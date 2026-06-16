# Embedding GWAS heritability and trait-annotation summary

## sam3
- Traits with h2 and GWAS summaries: 2048; traits with significant GWAS hits: 648.
- Median h2 for hit traits: 0.299; median h2 for no-hit traits: 0.316; Mann-Whitney one-sided p = 0.991.
- Spearman h2 vs hit indicator rho = -0.052, p = 0.0175; h2 vs number of significant markers rho = -0.067, p = 0.0026.

## dinov2
- Traits with h2 and GWAS summaries: 2048; traits with significant GWAS hits: 719.
- Median h2 for hit traits: 0.358; median h2 for no-hit traits: 0.348; Mann-Whitney one-sided p = 0.00186.
- Spearman h2 vs hit indicator rho = 0.064, p = 0.0037; h2 vs number of significant markers rho = 0.057, p = 0.00972.

## sam3 GWAS-signal groups
- Groups at Jaccard >= 0.25: 212; multi-trait groups: 87; singletons: 125.
- Median within-group abs Spearman correlation among multi-trait group embedding BLUEs: 0.387.
- Groups with top non-image/scored-trait correlation |rho| >= 0.3: 155; |rho| >= 0.4: 116.
- Top trait categories at |rho| >= 0.3: {'image_leaf_size': 93, 'disease_severity': 61, 'seed_color': 1}.
- Excluding NE image-derived traits, groups with top field-trait correlation |rho| >= 0.3: 76; |rho| >= 0.4: 24.
- Top field-trait categories at |rho| >= 0.3: {'architecture': 42, 'yield': 17, 'phenology': 12, 'seed_color': 5}.
- Examples:
  - group 72, n=140, within |rho|=0.348, top trait NE2025_image__percentUnhealthy (disease_severity), rho=0.429; examples embedding_mean_20;embedding_mean_74;embedding_mean_85;embedding_mean_129;embedding_mean_132;embedding_mean_141;embedding_mean_157;embedding_mean_165
  - group 55, n=37, within |rho|=0.328, top trait NE2025_image__leaf_pixels (image_leaf_size), rho=-0.361; examples embedding_mean_75;embedding_mean_153;embedding_mean_357;embedding_mean_595;embedding_mean_631;embedding_mean_714;embedding_mean_864;embedding_mean_988
  - group 31, n=27, within |rho|=0.387, top trait NE2025_image__human_score (disease_severity), rho=-0.643; examples embedding_mean_65;embedding_mean_275;embedding_mean_297;embedding_mean_570;embedding_mean_596;embedding_mean_629;embedding_mean_633;embedding_mean_807
  - group 33, n=21, within |rho|=0.360, top trait NE2025_image__human_score (disease_severity), rho=-0.351; examples embedding_mean_40;embedding_mean_123;embedding_mean_186;embedding_mean_332;embedding_mean_432;embedding_mean_494;embedding_mean_666;embedding_mean_674
  - group 74, n=15, within |rho|=0.438, top trait NE2025_image__log_leaf_pixels (image_leaf_size), rho=-0.861; examples embedding_mean_28;embedding_mean_277;embedding_mean_302;embedding_mean_373;embedding_mean_598;embedding_mean_680;embedding_mean_720;embedding_mean_855
- Top field-trait examples:
  - group 150, n=1, top field trait MI2020__leaf_width_cm (architecture), rho=0.509; examples embedding_mean_731
  - group 74, n=15, top field trait MI2020__leaf_width_cm (architecture), rho=-0.508; examples embedding_mean_28;embedding_mean_277;embedding_mean_302;embedding_mean_373;embedding_mean_598;embedding_mean_680;embedding_mean_720;embedding_mean_855
  - group 154, n=1, top field trait MI2020__leaf_width_cm (architecture), rho=0.484; examples embedding_mean_819
  - group 41, n=5, top field trait NE2025__leaf_number (architecture), rho=-0.470; examples embedding_mean_174;embedding_mean_792;embedding_mean_986;embedding_std_252;embedding_std_808
  - group 13, n=4, top field trait NE2020__leaf_width_cm (architecture), rho=0.468; examples embedding_mean_101;embedding_mean_126;embedding_mean_151;embedding_std_535

## dinov2 GWAS-signal groups
- Groups at Jaccard >= 0.25: 185; multi-trait groups: 77; singletons: 108.
- Median within-group abs Spearman correlation among multi-trait group embedding BLUEs: 0.359.
- Groups with top non-image/scored-trait correlation |rho| >= 0.3: 140; |rho| >= 0.4: 108.
- Top trait categories at |rho| >= 0.3: {'image_leaf_size': 112, 'disease_severity': 23, 'seed_color': 4, 'architecture': 1}.
- Excluding NE image-derived traits, groups with top field-trait correlation |rho| >= 0.3: 91; |rho| >= 0.4: 31.
- Top field-trait categories at |rho| >= 0.3: {'architecture': 67, 'phenology': 9, 'yield': 8, 'seed_color': 7}.
- Examples:
  - group 51, n=175, within |rho|=0.241, top trait NE2025_image__leaf_pixels (image_leaf_size), rho=0.793; examples dinov2_mean_41;dinov2_mean_52;dinov2_mean_53;dinov2_mean_56;dinov2_mean_62;dinov2_mean_73;dinov2_mean_92;dinov2_mean_100
  - group 37, n=54, within |rho|=0.196, top trait NE2025_image__log_leaf_pixels (image_leaf_size), rho=0.891; examples dinov2_mean_79;dinov2_mean_86;dinov2_mean_107;dinov2_mean_142;dinov2_mean_149;dinov2_mean_221;dinov2_mean_269;dinov2_mean_376
  - group 39, n=43, within |rho|=0.639, top trait NE2025_image__leaf_pixels (image_leaf_size), rho=-0.958; examples dinov2_mean_25;dinov2_mean_32;dinov2_mean_224;dinov2_mean_240;dinov2_mean_284;dinov2_mean_287;dinov2_mean_345;dinov2_mean_354
  - group 28, n=26, within |rho|=0.622, top trait NE2025_image__leaf_pixels (image_leaf_size), rho=-0.925; examples dinov2_mean_22;dinov2_mean_37;dinov2_mean_99;dinov2_mean_209;dinov2_mean_253;dinov2_mean_257;dinov2_mean_346;dinov2_mean_388
  - group 31, n=25, within |rho|=0.196, top trait NE2025_image__log_leaf_pixels (image_leaf_size), rho=-0.758; examples dinov2_mean_64;dinov2_mean_95;dinov2_mean_272;dinov2_mean_807;dinov2_mean_965;dinov2_mean_991;dinov2_std_199;dinov2_std_248
- Top field-trait examples:
  - group 39, n=43, top field trait MI2020__leaf_width_cm (architecture), rho=-0.509; examples dinov2_mean_25;dinov2_mean_32;dinov2_mean_224;dinov2_mean_240;dinov2_mean_284;dinov2_mean_287;dinov2_mean_345;dinov2_mean_354
  - group 48, n=2, top field trait MI2020__leaf_width_cm (architecture), rho=-0.495; examples dinov2_std_17;dinov2_std_506
  - group 77, n=7, top field trait MI2020__leaf_width_cm (architecture), rho=-0.492; examples dinov2_mean_28;dinov2_mean_335;dinov2_mean_375;dinov2_mean_382;dinov2_mean_577;dinov2_std_180;dinov2_std_470
  - group 28, n=26, top field trait NE2020__leaf_width_cm (architecture), rho=-0.489; examples dinov2_mean_22;dinov2_mean_37;dinov2_mean_99;dinov2_mean_209;dinov2_mean_253;dinov2_mean_257;dinov2_mean_346;dinov2_mean_388
  - group 2, n=6, top field trait NE2020__leaf_width_cm (architecture), rho=-0.488; examples dinov2_mean_974;dinov2_std_112;dinov2_std_171;dinov2_std_212;dinov2_std_239;dinov2_std_961
