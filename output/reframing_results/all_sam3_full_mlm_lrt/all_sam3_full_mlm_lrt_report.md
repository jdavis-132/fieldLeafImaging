# All SAM3 Full MLM LRT Report

## Model
- Samples: 895
- Markers: 4920234
- Traits: 2048
- PCs: 5
- Kinship: LOCO VanRaden
- Association: PANICLE multi-trait LOCO MLM with LRT refinement
- Effective marker number: 3169884
- Effective Bonferroni threshold: 1.577e-08

## Significant Signals
- Traits with at least one effective-threshold marker: 648/2048
- Total significant marker-trait associations: 13809
- Unique significant markers: 5555
- Significant loci after window merging: 319

## Top Traits By Significant Marker Count
| trait              | embedding_stat   |   embedding_index |   n_significant_effective_bonferroni |   n_q_lt_0_05_within_trait |       min_p |
|:-------------------|:-----------------|------------------:|-------------------------------------:|---------------------------:|------------:|
| embedding_mean_636 | mean             |               636 |                                 1112 |                      42633 | 1.38854e-11 |
| embedding_mean_645 | mean             |               645 |                                  953 |                       4715 | 8.05236e-14 |
| embedding_std_1020 | std              |              1020 |                                  552 |                       2771 | 6.52773e-14 |
| embedding_std_280  | std              |               280 |                                  530 |                       5321 | 7.59968e-14 |
| embedding_std_558  | std              |               558 |                                  516 |                       5910 | 1.79174e-12 |
| embedding_std_890  | std              |               890 |                                  492 |                       4192 | 1.99838e-11 |
| embedding_std_909  | std              |               909 |                                  436 |                       6104 | 5.38386e-11 |
| embedding_std_454  | std              |               454 |                                  436 |                       3451 | 4.24408e-11 |
| embedding_mean_241 | mean             |               241 |                                  386 |                       1787 | 1.45559e-12 |
| embedding_std_521  | std              |               521 |                                  368 |                       1962 | 1.95006e-12 |
| embedding_mean_335 | mean             |               335 |                                  307 |                       2671 | 1.08553e-11 |
| embedding_mean_585 | mean             |               585 |                                  294 |                       2336 | 5.96894e-12 |
| embedding_std_435  | std              |               435 |                                  276 |                       5872 | 3.36685e-11 |
| embedding_mean_530 | mean             |               530 |                                  265 |                       1502 | 1.28584e-11 |
| embedding_std_769  | std              |               769 |                                  258 |                       2647 | 7.53063e-11 |

## GWAS Signal Clustering
- Clustering uses binary sharing of significant loci, with markers merged into loci by chromosome and position window before computing Jaccard distances.
- Traits with any significant locus: 648/2048
- At Jaccard similarity >= 0.25: 87 multi-trait clusters; largest size 140.
- At Jaccard similarity >= 0.50: 87 multi-trait clusters; largest size 129.

## Largest Clusters, Similarity >= 0.25
|   cluster_jaccard_similarity_ge_0_25 |   n_traits |   median_significant_loci | example_traits                                                                                                                                          |
|-------------------------------------:|-----------:|--------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                   72 |        140 |                         1 | embedding_mean_20;embedding_mean_74;embedding_mean_85;embedding_mean_129;embedding_mean_132;embedding_mean_141;embedding_mean_157;embedding_mean_165    |
|                                   55 |         37 |                         1 | embedding_mean_75;embedding_mean_153;embedding_mean_357;embedding_mean_595;embedding_mean_631;embedding_mean_714;embedding_mean_864;embedding_mean_988  |
|                                   31 |         27 |                         1 | embedding_mean_65;embedding_mean_275;embedding_mean_297;embedding_mean_570;embedding_mean_596;embedding_mean_629;embedding_mean_633;embedding_mean_807  |
|                                   33 |         21 |                         2 | embedding_mean_40;embedding_mean_123;embedding_mean_186;embedding_mean_332;embedding_mean_432;embedding_mean_494;embedding_mean_666;embedding_mean_674  |
|                                   74 |         15 |                         1 | embedding_mean_28;embedding_mean_277;embedding_mean_302;embedding_mean_373;embedding_mean_598;embedding_mean_680;embedding_mean_720;embedding_mean_855  |
|                                   25 |         11 |                         1 | embedding_mean_30;embedding_mean_86;embedding_mean_173;embedding_mean_453;embedding_mean_466;embedding_mean_542;embedding_mean_717;embedding_mean_983   |
|                                   17 |         11 |                         1 | embedding_mean_559;embedding_mean_775;embedding_mean_904;embedding_mean_946;embedding_std_371;embedding_std_423;embedding_std_722;embedding_std_794     |
|                                   56 |         10 |                         1 | embedding_mean_508;embedding_mean_900;embedding_std_16;embedding_std_42;embedding_std_150;embedding_std_245;embedding_std_642;embedding_std_827         |
|                                   10 |         10 |                         1 | embedding_mean_127;embedding_mean_339;embedding_mean_520;embedding_mean_682;embedding_mean_833;embedding_mean_925;embedding_mean_995;embedding_std_67   |
|                                    7 |         10 |                         1 | embedding_mean_119;embedding_mean_441;embedding_mean_599;embedding_mean_762;embedding_mean_890;embedding_std_212;embedding_std_228;embedding_std_538    |
|                                   46 |         10 |                         8 | embedding_mean_300;embedding_mean_474;embedding_mean_514;embedding_mean_619;embedding_mean_636;embedding_mean_739;embedding_mean_1003;embedding_std_117 |
|                                   62 |          8 |                         1 | embedding_mean_408;embedding_mean_978;embedding_std_191;embedding_std_352;embedding_std_375;embedding_std_701;embedding_std_774;embedding_std_847       |
|                                   67 |          7 |                         2 | embedding_mean_164;embedding_mean_454;embedding_mean_510;embedding_mean_549;embedding_mean_585;embedding_mean_927;embedding_std_679                     |
|                                   28 |          6 |                         1 | embedding_mean_29;embedding_std_163;embedding_std_484;embedding_std_527;embedding_std_681;embedding_std_939                                             |
|                                   44 |          6 |                         1 | embedding_mean_99;embedding_mean_273;embedding_mean_660;embedding_mean_780;embedding_std_849;embedding_std_863                                          |
|                                    5 |          5 |                         1 | embedding_mean_234;embedding_mean_469;embedding_std_77;embedding_std_487;embedding_std_568                                                              |
|                                   41 |          5 |                         1 | embedding_mean_174;embedding_mean_792;embedding_mean_986;embedding_std_252;embedding_std_808                                                            |
|                                   65 |          5 |                         1 | embedding_mean_107;embedding_std_149;embedding_std_305;embedding_std_374;embedding_std_927                                                              |
|                                   27 |          5 |                         1 | embedding_mean_399;embedding_std_222;embedding_std_572;embedding_std_655;embedding_std_851                                                              |
|                                   52 |          5 |                         1 | embedding_mean_359;embedding_std_331;embedding_std_336;embedding_std_578;embedding_std_969                                                              |

## Largest Clusters, Similarity >= 0.50
|   cluster_jaccard_similarity_ge_0_50 |   n_traits |   median_significant_loci | example_traits                                                                                                                                         |
|-------------------------------------:|-----------:|--------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                  111 |        129 |                       1   | embedding_mean_20;embedding_mean_74;embedding_mean_85;embedding_mean_129;embedding_mean_132;embedding_mean_141;embedding_mean_157;embedding_mean_165   |
|                                   80 |         28 |                       1   | embedding_mean_75;embedding_mean_153;embedding_mean_357;embedding_mean_595;embedding_mean_631;embedding_mean_714;embedding_mean_864;embedding_mean_988 |
|                                   44 |         27 |                       1   | embedding_mean_65;embedding_mean_275;embedding_mean_297;embedding_mean_570;embedding_mean_596;embedding_mean_629;embedding_mean_633;embedding_mean_807 |
|                                  122 |         15 |                       1   | embedding_mean_28;embedding_mean_277;embedding_mean_302;embedding_mean_373;embedding_mean_598;embedding_mean_680;embedding_mean_720;embedding_mean_855 |
|                                   25 |         10 |                       1   | embedding_mean_559;embedding_mean_775;embedding_mean_904;embedding_mean_946;embedding_std_371;embedding_std_423;embedding_std_722;embedding_std_910    |
|                                   36 |         10 |                       1   | embedding_mean_30;embedding_mean_86;embedding_mean_173;embedding_mean_453;embedding_mean_466;embedding_mean_542;embedding_mean_717;embedding_mean_983  |
|                                    9 |          8 |                       1   | embedding_mean_119;embedding_mean_762;embedding_mean_890;embedding_std_212;embedding_std_228;embedding_std_538;embedding_std_620;embedding_std_803     |
|                                   15 |          8 |                       1   | embedding_mean_127;embedding_mean_339;embedding_mean_682;embedding_mean_925;embedding_mean_995;embedding_std_67;embedding_std_574;embedding_std_746    |
|                                   50 |          8 |                       3   | embedding_mean_40;embedding_mean_123;embedding_mean_742;embedding_mean_942;embedding_std_230;embedding_std_288;embedding_std_633;embedding_std_873     |
|                                   93 |          7 |                       1   | embedding_mean_408;embedding_std_191;embedding_std_352;embedding_std_375;embedding_std_701;embedding_std_774;embedding_std_847                         |
|                                   87 |          7 |                       1   | embedding_mean_508;embedding_mean_900;embedding_std_16;embedding_std_42;embedding_std_150;embedding_std_642;embedding_std_827                          |
|                                   62 |          6 |                       1   | embedding_mean_99;embedding_mean_273;embedding_mean_660;embedding_mean_780;embedding_std_849;embedding_std_863                                         |
|                                   46 |          6 |                       1   | embedding_mean_332;embedding_mean_432;embedding_mean_666;embedding_std_291;embedding_std_340;embedding_std_944                                         |
|                                  103 |          6 |                       1.5 | embedding_mean_164;embedding_mean_454;embedding_mean_510;embedding_mean_549;embedding_mean_585;embedding_mean_927                                      |
|                                   40 |          6 |                       1   | embedding_mean_29;embedding_std_163;embedding_std_484;embedding_std_527;embedding_std_681;embedding_std_939                                            |
|                                   39 |          5 |                       1   | embedding_mean_399;embedding_std_222;embedding_std_572;embedding_std_655;embedding_std_851                                                             |
|                                   83 |          5 |                       2   | embedding_std_144;embedding_std_171;embedding_std_309;embedding_std_601;embedding_std_1021                                                             |
|                                   76 |          5 |                       1   | embedding_mean_359;embedding_std_331;embedding_std_336;embedding_std_578;embedding_std_969                                                             |
|                                   59 |          5 |                       1   | embedding_mean_174;embedding_mean_792;embedding_mean_986;embedding_std_252;embedding_std_808                                                           |
|                                    7 |          5 |                       1   | embedding_mean_234;embedding_mean_469;embedding_std_77;embedding_std_487;embedding_std_568                                                             |

## Interpretation Placeholder
Use this section to interpret whether shared loci form coherent embedding modules. Large clusters support shared genetic control across embedding axes; smaller clusters or singleton traits support genetically distinct symptom/image axes.
