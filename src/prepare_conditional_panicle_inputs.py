#!/usr/bin/env python3
"""Prepare PANICLE inputs for severity-conditioned embedding GWAS.

The source VCF is assumed to have already had slow global genotype filters
applied, including the heterozygosity filter described in the manuscript.
This script only prepares trait-specific sample, phenotype, covariate, and
command files for downstream PANICLE MLM runs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VCF = Path("/home/james/projects/SorghumDataCleanup/sorghum_925genotypes_filtered_v3.vcf.gz")
OUT_DIR = ROOT / "output" / "conditional_panicle"


def feature_to_embedding_column(feature: int) -> str:
    if feature < 1024:
        return f"embedding_mean_{feature}"
    return f"embedding_std_{feature - 1024}"


def get_vcf_samples(vcf: Path) -> set[str]:
    result = subprocess.run(
        ["bcftools", "query", "-l", str(vcf)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return set(result.stdout.splitlines())


def selected_sam3_embedding_columns() -> pd.DataFrame:
    fi = pd.read_csv(ROOT / "data" / "rf" / "sam3_human_scores_embedding_feature_importances_rf.csv")
    values = fi.drop(columns=["Unnamed: 0"], errors="ignore")
    mean_fi = values.mean(axis=0)
    selected = mean_fi[mean_fi > 0.003].sort_values(ascending=False)
    rows = []
    for feature_name, mean_importance in selected.items():
        feature = int(feature_name)
        rows.append(
            {
                "feature_index": feature,
                "trait": feature_to_embedding_column(feature),
                "embedding_stat": "mean" if feature < 1024 else "std",
                "embedding_number": feature if feature < 1024 else feature - 1024,
                "mean_feature_importance": mean_importance,
            }
        )
    return pd.DataFrame(rows)


def write_dataset(
    name: str,
    ne: pd.DataFrame,
    trait_cols: list[str],
    covariate_cols: list[str],
    vcf_samples: set[str],
) -> dict[str, object]:
    dataset_dir = OUT_DIR / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    required = ["genotype", *trait_cols, *covariate_cols]
    data = ne.loc[ne["genotype"].isin(vcf_samples), required].copy()
    data = data.dropna(subset=covariate_cols)

    phenotype = data[["genotype", *trait_cols]]
    covariates = data[["genotype", *covariate_cols]]
    samples = phenotype["genotype"].drop_duplicates().sort_values()

    phenotype_path = dataset_dir / f"{name}_embedding_phenotypes.csv"
    covariate_path = dataset_dir / f"{name}_severity_covariates.csv"
    sample_path = dataset_dir / f"{name}_samples.txt"
    traits_path = dataset_dir / f"{name}_traits.txt"

    phenotype.to_csv(phenotype_path, index=False)
    covariates.to_csv(covariate_path, index=False)
    samples.to_csv(sample_path, index=False, header=False)
    pd.Series(trait_cols).to_csv(traits_path, index=False, header=False)

    return {
        "dataset": name,
        "n_samples": int(samples.shape[0]),
        "n_traits": len(trait_cols),
        "covariates": "+".join(covariate_cols),
        "phenotype_file": str(phenotype_path.relative_to(ROOT)),
        "covariate_file": str(covariate_path.relative_to(ROOT)),
        "sample_file": str(sample_path.relative_to(ROOT)),
        "traits_file": str(traits_path.relative_to(ROOT)),
    }


def write_command_files(vcf: Path, summaries: list[dict[str, object]]) -> None:
    commands = []
    for summary in summaries:
        name = str(summary["dataset"])
        dataset_dir = OUT_DIR / name
        sample_file = dataset_dir / f"{name}_samples.txt"
        filtered_vcf = dataset_dir / f"{name}.vcf.gz"
        commands.append(
            "bcftools view "
            "-S {sample_file} "
            "-r 1,2,3,4,5,6,7,8,9,10 "
            "-m2 -M2 "
            "{vcf} -Ou | "
            "bcftools view -i 'F_MISSING<=0.70' -Ou | "
            "bcftools +fill-tags -Ou -- -t MAF | "
            "bcftools view -i 'MAF>=0.05' -Oz -o {filtered_vcf}\n"
            "tabix -p vcf {filtered_vcf}"
            .format(
                sample_file=sample_file,
                vcf=vcf,
                filtered_vcf=filtered_vcf,
            )
        )

    command_path = OUT_DIR / "filter_vcf_commands.sh"
    command_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n\n".join(commands) + "\n")
    command_path.chmod(0o755)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vcf_samples = get_vcf_samples(DEFAULT_VCF)

    blues = pd.read_csv(ROOT / "data" / "blues_all.csv")
    ne = blues.loc[blues["location"] == "NE"].copy()
    selected = selected_sam3_embedding_columns()
    trait_cols = selected["trait"].tolist()
    selected.to_csv(OUT_DIR / "selected_sam3_embedding_traits.csv", index=False)

    summaries = [
        write_dataset("condition_exg", ne, trait_cols, ["percentUnhealthy"], vcf_samples),
        write_dataset("condition_human_exg", ne, trait_cols, ["human_score", "percentUnhealthy"], vcf_samples),
    ]
    pd.DataFrame(summaries).to_csv(OUT_DIR / "input_summary.csv", index=False)
    write_command_files(DEFAULT_VCF, summaries)

    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"\nWrote inputs to {OUT_DIR}")
    print(f"Run {OUT_DIR / 'filter_vcf_commands.sh'} to create trait-specific VCFs.")


if __name__ == "__main__":
    main()
