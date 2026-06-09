#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(lme4)
})

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx)) {
    return(default)
  }
  if (idx == length(args)) {
    stop(str_c("Missing value for ", flag))
  }
  args[[idx + 1]]
}

embeddings_path <- get_arg("--embeddings", "/home/james/leaf_imaging/dinov2_20260522.csv")
field_index_path <- get_arg("--field-index", "data/ne2025/SbDiv_ne2025_fieldindex.csv")
genotype_alignment_path <- get_arg("--genotype-alignment", "data/ne2025/genotype_alignment_reseq.csv")
images_keep_path <- get_arg("--images-keep", "data/ne2025/images_keep_all.csv")
out_dir <- get_arg("--out-dir", "output/dinov2_20260522_heritability")
write_plot_means <- get_arg("--write-plot-means", "false") %in% c("true", "TRUE", "1", "yes")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

partition_variance <- function(df, response, label, model_statement) {
  lm_formula <- as.formula(paste(response, model_statement))
  model <- lmer(
    lm_formula,
    data = df,
    na.action = na.omit,
    control = lmerControl(check.conv.singular = "ignore")
  )
  vc <- as.data.frame(VarCorr(model), row.names = TRUE, order = "cov.last", comp = "Variance") %>%
    as_tibble() %>%
    mutate(responseVar = response)
  total_var <- sum(vc$vcov)
  vc %>%
    rowwise() %>%
    mutate(
      pctVar = vcov / total_var * 100,
      label = label
    ) %>%
    dplyr::select(responseVar, grp, vcov, pctVar, label)
}

crop_image_id <- function(image_path) {
  basename(image_path) %>%
    str_remove("-05_00_[0-9]\\.(png|npz)$") %>%
    str_remove("-05_00\\.jpg$") %>%
    str_remove("-05_00$")
}

message("Reading DINOv2 embeddings: ", embeddings_path)
embeddings <- read_csv(embeddings_path, show_col_types = FALSE)
trait_cols <- colnames(embeddings)[str_detect(colnames(embeddings), "^(mean|std)_[0-9]+$")]
if (length(trait_cols) == 0) {
  stop("No DINOv2 mean_/std_ trait columns found.")
}

message("Preparing plot-level means for ", length(trait_cols), " traits")
keep <- read_tsv(images_keep_path, col_names = c("image_id"), skip = 1, show_col_types = FALSE)
field <- read_csv(field_index_path, show_col_types = FALSE) %>%
  rename_with(str_trim) %>%
  mutate(genotype = str_remove_all(genotype, " "))
alignment <- read_csv(genotype_alignment_path, show_col_types = FALSE) %>%
  rename(genotype_markers = genotype_reseq) %>%
  mutate(genotype_idx = str_remove_all(genotype_idx, " ")) %>%
  distinct()

plot_means <- embeddings %>%
  filter(!str_detect(image_path, "cropped_transparent_bg")) %>%
  mutate(
    plotNumber = basename(image_path) %>% str_split_i("_", 1) %>% as.numeric(),
    image_id = crop_image_id(image_path)
  ) %>%
  filter(image_id %in% keep$image_id) %>%
  select(image_path, plotNumber, image_id, all_of(trait_cols)) %>%
  left_join(field, join_by(plotNumber), relationship = "many-to-one") %>%
  filter(!genotype %in% c("Border", "Check", "Fill", "Mixed")) %>%
  left_join(alignment, join_by(genotype == genotype_idx)) %>%
  mutate(genotype_markers = case_when(is.na(genotype_markers) ~ genotype, .default = genotype_markers)) %>%
  filter(!is.na(genotype_markers)) %>%
  select(!c(genotype, image_path, image_id)) %>%
  rename(genotype = genotype_markers) %>%
  group_by(range, row, genotype, plotNumber) %>%
  summarise(across(all_of(trait_cols), ~ mean(.x, na.rm = TRUE)), .groups = "drop")

if (write_plot_means) {
  write_csv(plot_means, file.path(out_dir, "dinov2_20260522_plot_means.csv"))
}

variance_rows <- list()
heritability_rows <- list()
model_statement <- "~ (1|range) + (1|row) + (1|genotype)"

for (i in seq_along(trait_cols)) {
  trait <- trait_cols[[i]]
  if (i %% 50 == 1 || i == length(trait_cols)) {
    message("[", i, "/", length(trait_cols), "] ", trait)
  }
  result <- tryCatch(
    {
      vp <- partition_variance(plot_means, trait, trait, model_statement)
      residual_var <- vp$vcov[vp$grp == "Residual"]
      genotype_var <- vp$vcov[vp$grp == "genotype"]
      phenotypic_var <- sum(vp$vcov) - residual_var / 2
      h2 <- genotype_var / phenotypic_var
      vp$broad_sense_h2 <- h2
      vp$n_plot_means <- nrow(plot_means)
      vp$n_genotypes <- n_distinct(plot_means$genotype)
      vp$status <- "ok"
      list(
        variance = vp,
        summary = tibble(
          trait = trait,
          trait_blues_all_name = str_replace(trait, "^mean_", "dinov2_mean_") %>%
            str_replace("^std_", "dinov2_std_"),
          broad_sense_h2 = h2,
          genotype_vcov = genotype_var,
          residual_vcov = residual_var,
          phenotypic_v_for_h2 = phenotypic_var,
          n_plot_means = nrow(plot_means),
          n_genotypes = n_distinct(plot_means$genotype),
          status = "ok",
          error = NA_character_
        )
      )
    },
    error = function(e) {
      list(
        variance = tibble(
          responseVar = trait,
          grp = NA_character_,
          vcov = NA_real_,
          pctVar = NA_real_,
          label = trait,
          broad_sense_h2 = NA_real_,
          n_plot_means = nrow(plot_means),
          n_genotypes = n_distinct(plot_means$genotype),
          status = "error"
        ),
        summary = tibble(
          trait = trait,
          trait_blues_all_name = str_replace(trait, "^mean_", "dinov2_mean_") %>%
            str_replace("^std_", "dinov2_std_"),
          broad_sense_h2 = NA_real_,
          genotype_vcov = NA_real_,
          residual_vcov = NA_real_,
          phenotypic_v_for_h2 = NA_real_,
          n_plot_means = nrow(plot_means),
          n_genotypes = n_distinct(plot_means$genotype),
          status = "error",
          error = conditionMessage(e)
        )
      )
    }
  )
  variance_rows[[i]] <- result$variance
  heritability_rows[[i]] <- result$summary
}

variance_components <- bind_rows(variance_rows)
heritability <- bind_rows(heritability_rows)

variance_path <- file.path(out_dir, "dinov2_20260522_variance_components.csv")
h2_path <- file.path(out_dir, "dinov2_20260522_heritability.csv")
summary_path <- file.path(out_dir, "dinov2_20260522_heritability_summary.csv")

write_csv(variance_components, variance_path)
write_csv(heritability, h2_path)

summary <- heritability %>%
  summarise(
    n_traits = n(),
    n_ok = sum(status == "ok"),
    n_error = sum(status != "ok"),
    min_h2 = min(broad_sense_h2, na.rm = TRUE),
    median_h2 = median(broad_sense_h2, na.rm = TRUE),
    mean_h2 = mean(broad_sense_h2, na.rm = TRUE),
    max_h2 = max(broad_sense_h2, na.rm = TRUE),
    n_plot_means = first(n_plot_means),
    n_genotypes = first(n_genotypes),
    model = model_statement
  )
write_csv(summary, summary_path)

print(summary)
message("Wrote ", h2_path)
message("Wrote ", variance_path)
