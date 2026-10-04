# candidates — artifact-derived repair tables

These tables render stored repair outputs. File existence alone does not
establish formal completion; consult STATUS.json and the acceptance report.

## audit_run_20261003_Cslot_audit4/availability_intersection.csv

Source: `results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/availability_intersection.csv`; SHA256 `80075c4cc2eae1ba10368883c6838fa037cdf71d4d119d3d4f218db074c6fb46`.

| family | cell | candidate_source | source_path | k | cell_kind | same_category_proposals_required | evaluation_universe_rows | historical_old_available_rows | historical_evaluated_rows | historical_candidate_available_rows | corrected_true_category_available_rows | old_new_common_rows | old_only_rows_lost_under_corrected_category | corrected_only_rows_not_in_historical_cohort | historical_rows_lost_under_corrected_category | old_common_retention | new_common_retention | historical_retention | mismatched_target_proposal_rows | old_variant | new_variant | candidate_variant | comparison_cohort |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RPN | hard5 | manifest_candidate_supply_universe | cache/manifests*/same_category_test{A,B}.jsonl | 5 | hard | 4 | 10425 | 9623 |  |  | 9626 | 9602 | 21 | 24 |  | 0.9978177283591395 | 0.9975067525451901 |  |  |  | true_target_category_v1 |  | old_new_common_availability_intersection |
| RPN | hard10 | manifest_candidate_supply_universe | cache/manifests*/same_category_test{A,B}.jsonl | 10 | hard | 9 | 10425 | 6874 |  |  | 6880 | 6856 | 18 | 24 |  | 0.9973814372999709 | 0.9965116279069768 |  |  |  | true_target_category_v1 |  | old_new_common_availability_intersection |
| RPN | expb_m0 | manifest_candidate_supply_universe | cache/manifests*/same_category_test{A,B}.jsonl | 10 | dose | 0 | 10425 | 7528 |  |  | 10425 | 7528 | 0 | 2897 |  | 1.0 | 0.7221103117505995 |  |  |  | true_target_category_v1 |  | old_new_common_availability_intersection |
| RPN | expb_m2 | manifest_candidate_supply_universe | cache/manifests*/same_category_test{A,B}.jsonl | 10 | dose | 2 | 10425 | 7528 |  |  | 10187 | 7528 | 0 | 2659 |  | 1.0 | 0.7389810542848729 |  |  |  | true_target_category_v1 |  | old_new_common_availability_intersection |
| RPN | expb_m4 | manifest_candidate_supply_universe | cache/manifests*/same_category_test{A,B}.jsonl | 10 | dose | 4 | 10425 | 7528 |  |  | 9626 | 7516 | 12 | 2110 |  | 0.9984059511158342 | 0.7808019945979638 |  |  |  | true_target_category_v1 |  | old_new_common_availability_intersection |
| RPN | expb_m8 | manifest_candidate_supply_universe | cache/manifests*/same_category_test{A,B}.jsonl | 10 | dose | 8 | 10425 | 7528 |  |  | 7534 | 7507 | 21 | 27 |  | 0.9972104144527099 | 0.9964162463498806 |  |  |  | true_target_category_v1 |  | old_new_common_availability_intersection |
| DETR | hard5 | manifest_candidate_supply_universe | cache/manifests*/same_category_test{A,B}.jsonl | 5 | hard | 4 | 10601 | 8376 |  |  | 8396 | 8373 | 3 | 23 |  | 0.9996418338108882 | 0.9972606002858504 |  |  |  | true_target_category_v1 |  | old_new_common_availability_intersection |
| RPN | hard5 | phase1f_frozen_candidate_archive | results/phase1f_hard_semantic/manifests/hard5_candidates.npz | 5 | hard |  |  | 9487 | 9487 |  | 9466 | 9466 | 21 | 0 | 21 | 0.9977864446084115 | 1.0 | 0.9977864446084115 | 54 |  |  | true_target_category_v1 | historical_source_old_new_common_intersection |
| RPN | hard10 | phase1f_frozen_candidate_archive | results/phase1f_hard_semantic/manifests/hard10_candidates.npz | 10 | hard |  |  | 6765 | 6765 |  | 6747 | 6747 | 18 | 0 | 18 | 0.9973392461197339 | 1.0 | 0.9973392461197339 | 24 |  |  | true_target_category_v1 | historical_source_old_new_common_intersection |
| RPN | expb_m0 | phase1f_frozen_candidate_archive | results/phase1f_hard_semantic/manifests/expb_m0_candidates.npz | 10 | dose |  |  | 7410 | 7410 |  | 7410 | 7410 | 0 | 0 | 0 | 1.0 | 1.0 | 1.0 | 27 |  |  | true_target_category_v1 | historical_source_old_new_common_intersection |
| RPN | expb_m2 | phase1f_frozen_candidate_archive | results/phase1f_hard_semantic/manifests/expb_m2_candidates.npz | 10 | dose |  |  | 7410 | 7410 |  | 7410 | 7410 | 0 | 0 | 0 | 1.0 | 1.0 | 1.0 | 27 |  |  | true_target_category_v1 | historical_source_old_new_common_intersection |
| RPN | expb_m4 | phase1f_frozen_candidate_archive | results/phase1f_hard_semantic/manifests/expb_m4_candidates.npz | 10 | dose |  |  | 7410 | 7410 |  | 7398 | 7398 | 12 | 0 | 12 | 0.9983805668016195 | 1.0 | 0.9983805668016195 | 27 |  |  | true_target_category_v1 | historical_source_old_new_common_intersection |
| RPN | expb_m8 | phase1f_frozen_candidate_archive | results/phase1f_hard_semantic/manifests/expb_m8_candidates.npz | 10 | dose |  |  | 7410 | 7410 |  | 7389 | 7389 | 21 | 0 | 21 | 0.9971659919028341 | 1.0 | 0.9971659919028341 | 27 |  |  | true_target_category_v1 | historical_source_old_new_common_intersection |
| RPN | hard5 | v2_p1_frozen_hard_k5_predictions | results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed1.npz | 5 | hard |  |  | 9623 | 9623 |  | 9602 | 9602 | 21 | 0 | 21 | 0.9978177283591395 | 1.0 | 0.9978177283591395 | 54 |  |  | true_target_category_v1 | historical_source_old_new_common_intersection |
| DETR | hard5 | v2_p1_frozen_hard_k5_predictions | results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed1.npz | 5 | hard |  |  | 8376 | 8376 |  | 8373 | 8373 | 3 | 0 | 3 | 0.9996418338108882 | 1.0 | 0.9996418338108882 | 11 |  |  | true_target_category_v1 | historical_source_old_new_common_intersection |

## audit_run_20261003_Cslot_audit4/forward/pilot_sensitivity_one_rep_20261003_01/dose_arm_point_preflight.csv

Source: `results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/forward/pilot_sensitivity_one_rep_20261003_01/dose_arm_point_preflight.csv`; SHA256 `cd4696f99e572c117ba710f6918fffa26b7b9b4d3290b6b7af2f95ec6856f222`.

| group | family | candidate_source | eval_split | dose_level | metric | confidence_head | units | old_m_minus_old_m0_point | corrected_m_minus_corrected_m0_point | corrected_m_minus_fixed_original_old_m0_point | version_change_dod_point | identity_residual | n_rows | n_images | n_replicates | raw_replicates_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m2 | accuracy | msp | proportion difference | -0.1417873415437362 | -0.14174222944015882 | -0.1418324536473136 | -4.511210357738271e-05 | 0.0 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m2 | auroc_correct | msp | AUROC difference | -0.06304086464067127 | -0.06303016383640718 | -0.06312700958629984 | -1.0700804264092886e-05 | 4.6264939579029885e-18 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m2 | auroc_correct | stats | AUROC difference | -0.062370684896528306 | -0.06239644055521983 | -0.062435127866006725 | 2.5755658691524925e-05 | 2.3140940118987485e-18 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m2 | auroc_correct | e1b | AUROC difference | -0.05104209263726719 | -0.05110327737978482 | -0.05105061609508613 | 6.118474251763188e-05 | -2.3174821436877657e-18 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m2 | e_aurc | msp | eAURC difference | 0.05513132458430505 | 0.055129914948472025 | 0.05516630961686613 | 1.4096358330236332e-06 | 2.3130352207146807e-18 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m2 | e_aurc | stats | eAURC difference | 0.05471285706485531 | 0.05471762135659739 | 0.05473550399530654 | -4.764291742074637e-06 | -4.625646924955734e-18 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m2 | e_aurc | e1b | eAURC difference | 0.04966089425979555 | 0.04966591230394427 | 0.04965811147813757 | -5.018044148726493e-06 | 4.625646924955734e-18 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m2 | rer_at_50 | msp | RER ratio difference | -0.28018322617435654 | -0.2797497669410654 | -0.2796735211048754 | -0.0004334592332911706 | 3.7025504190379976e-17 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m2 | rer_at_50 | stats | RER ratio difference | -0.2764774721934215 | -0.2762690023749728 | -0.2761931163891645 | -0.0002084698184486681 | -1.8512752095189988e-17 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m2 | rer_at_50 | e1b | RER ratio difference | -0.24239082993028216 | -0.2424181541795265 | -0.2423369475775533 | 2.7324249244342624e-05 | 0.0 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m2 | rer_at_80 | msp | RER ratio difference | -0.15718187094809158 | -0.15708578113504182 | -0.15751283264502633 | -9.608981304976931e-05 | 9.256376047594994e-18 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m2 | rer_at_80 | stats | RER ratio difference | -0.16318292017294014 | -0.1627275301024597 | -0.16337237410207076 | -0.0004553900704804455 | -9.269928574751063e-18 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m2 | rer_at_80 | e1b | RER ratio difference | -0.14675567140570972 | -0.1470203108925051 | -0.14722857040304296 | 0.000264639486795402 | -9.269928574751063e-18 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m4 | accuracy | msp | proportion difference | -0.17783191230207063 | -0.17778680019849324 | -0.177877024405648 | -4.511210357738271e-05 | 0.0 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m4 | auroc_correct | msp | AUROC difference | -0.05072295423982365 | -0.05063043187004782 | -0.05072727761994048 | -9.252236977582336e-05 | -2.3174821436877657e-18 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m4 | auroc_correct | stats | AUROC difference | -0.04914369668137505 | -0.04914720005164889 | -0.04918588736243579 | 3.5033702738459147e-06 | -4.626070441429361e-18 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m4 | auroc_correct | e1b | AUROC difference | -0.030112490802235176 | -0.03018916451374143 | -0.03013650322904275 | 7.66737115062514e-05 | 1.1519648082658485e-18 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m4 | e_aurc | msp | eAURC difference | 0.05625727627597687 | 0.05621906960773541 | 0.05625546427612951 | 3.82066682414565e-05 | 4.628188023797497e-18 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m4 | e_aurc | stats | eAURC difference | 0.05527491367890671 | 0.05527689124109129 | 0.05529477387980044 | -1.9775621845815237e-06 | 4.626070441429361e-18 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m4 | e_aurc | e1b | eAURC difference | 0.04633332867539348 | 0.04635536805449154 | 0.046347567228684834 | -2.2039379098059975e-05 | 0.0 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m4 | rer_at_50 | msp | RER ratio difference | -0.28802876696173213 | -0.2882600254216949 | -0.2881837795855049 | 0.000231258459962691 | 5.551115123125783e-17 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m4 | rer_at_50 | stats | RER ratio difference | -0.286382117084593 | -0.286198844048063 | -0.2861229580622547 | -0.00018327303653000357 | 1.8512752095189988e-17 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m4 | rer_at_50 | e1b | RER ratio difference | -0.23678923716997927 | -0.2372389854676531 | -0.2371577788656799 | 0.0004497482976738206 | 9.269928574751063e-18 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m4 | rer_at_80 | msp | RER ratio difference | -0.16020492111531534 | -0.15995242899031967 | -0.16037948050030415 | -0.0002524921249956828 | 9.269928574751063e-18 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m4 | rer_at_80 | stats | RER ratio difference | -0.1639140162391357 | -0.1633134604028934 | -0.16395830440250447 | -0.0006005558362423002 | 0.0 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m4 | rer_at_80 | e1b | RER ratio difference | -0.14849574018363798 | -0.1483335922283734 | -0.1485418517389112 | -0.00016214795526459103 | 0.0 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m8 | accuracy | msp | proportion difference | -0.1989443767762891 | -0.19898948887986648 | -0.19907971308702124 | 4.511210357738271e-05 | 0.0 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m8 | auroc_correct | msp | AUROC difference | -0.02514536475329317 | -0.02518367498986081 | -0.02528052073975347 | 3.831023656764021e-05 | 0.0 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m8 | auroc_correct | stats | AUROC difference | -0.023350238364384362 | -0.02343360033455481 | -0.023472287645341705 | 8.3361970170448e-05 | 0.0 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m8 | auroc_correct | e1b | AUROC difference | 0.007384348956776532 | 0.007227134797562947 | 0.007279796082261629 | 0.00015721415921358486 | 2.981555974335137e-19 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m8 | e_aurc | msp | eAURC difference | 0.04523264883042553 | 0.045314042427619626 | 0.045350437096013725 | -8.139359719409416e-05 | -4.621411760219463e-18 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m8 | e_aurc | stats | eAURC difference | 0.04371113202953144 | 0.0437961495808335 | 0.043814032219542644 | -8.501755130206161e-05 | 2.3174821436877657e-18 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m8 | e_aurc | e1b | eAURC difference | 0.029401808248143712 | 0.02949768794571427 | 0.029489887119907566 | -9.58796975705593e-05 | 2.3174821436877657e-18 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m8 | rer_at_50 | msp | RER ratio difference | -0.2518638256820611 | -0.2521960416909292 | -0.25211979585473926 | 0.0003322160088680877 | -1.848564704087785e-17 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m8 | rer_at_50 | stats | RER ratio difference | -0.2509201876784743 | -0.2514490921888753 | -0.25137320620306697 | 0.000528904510400959 | 3.69712940817557e-17 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m8 | rer_at_50 | e1b | RER ratio difference | -0.18281348237625814 | -0.18336373348299986 | -0.18328252688102667 | 0.0005502511067417467 | -1.8539857149502126e-17 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m8 | rer_at_80 | msp | RER ratio difference | -0.15871765854284736 | -0.15841921151880875 | -0.15884626302879323 | -0.00029844702403863943 | 2.7755575615628914e-17 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m8 | rer_at_80 | stats | RER ratio difference | -0.16171420021074812 | -0.16119759719197527 | -0.16184244119158633 | -0.0005166030187728488 | 0.0 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__pooled_testA_testB | RPN | phase1f_frozen_candidate_archive | pooled_testA_testB | expb_m8 | rer_at_80 | e1b | RER ratio difference | -0.13490566139045745 | -0.13483423040439976 | -0.13504248991493759 | -7.143098605771891e-05 | 2.7755575615628914e-17 | 7389 | 1188 | 1 | group_780e07ee900c5f2c_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m2 | accuracy | msp | proportion difference | -0.12866852023478526 | -0.12866852023478526 | -0.12866852023478526 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m2 | auroc_correct | msp | AUROC difference | -0.06560352358854608 | -0.06560352358854608 | -0.06560352358854608 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m2 | auroc_correct | stats | AUROC difference | -0.06343277369782996 | -0.06343277369782996 | -0.06343277369782996 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m2 | auroc_correct | e1b | AUROC difference | -0.0523252135639987 | -0.0523252135639987 | -0.0523252135639987 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m2 | e_aurc | msp | eAURC difference | 0.04753389409879144 | 0.04753389409879144 | 0.04753389409879144 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m2 | e_aurc | stats | eAURC difference | 0.047075533255010106 | 0.047075533255010106 | 0.047075533255010106 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m2 | e_aurc | e1b | eAURC difference | 0.043313190806019884 | 0.043313190806019884 | 0.043313190806019884 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m2 | rer_at_50 | msp | RER ratio difference | -0.26394858602548216 | -0.26394858602548216 | -0.26394858602548216 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m2 | rer_at_50 | stats | RER ratio difference | -0.2553760770019015 | -0.2553760770019015 | -0.2553760770019015 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m2 | rer_at_50 | e1b | RER ratio difference | -0.2246890295145637 | -0.2246890295145637 | -0.2246890295145637 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m2 | rer_at_80 | msp | RER ratio difference | -0.1967589783574076 | -0.1967589783574076 | -0.1967589783574076 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m2 | rer_at_80 | stats | RER ratio difference | -0.19461418308210474 | -0.19461418308210474 | -0.19461418308210474 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m2 | rer_at_80 | e1b | RER ratio difference | -0.17379462555286587 | -0.17379462555286587 | -0.17379462555286587 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m4 | accuracy | msp | proportion difference | -0.16295953042940992 | -0.16295953042940992 | -0.16295953042940992 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m4 | auroc_correct | msp | AUROC difference | -0.05827844037643944 | -0.05827844037643944 | -0.05827844037643944 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m4 | auroc_correct | stats | AUROC difference | -0.055848014939758474 | -0.055848014939758474 | -0.055848014939758474 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m4 | auroc_correct | e1b | AUROC difference | -0.035769850176991635 | -0.035769850176991635 | -0.035769850176991635 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m4 | e_aurc | msp | eAURC difference | 0.050845349447461295 | 0.050845349447461295 | 0.050845349447461295 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m4 | e_aurc | stats | eAURC difference | 0.050212640600110416 | 0.050212640600110416 | 0.050212640600110416 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m4 | e_aurc | e1b | eAURC difference | 0.04241703871486297 | 0.04241703871486297 | 0.04241703871486297 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m4 | rer_at_50 | msp | RER ratio difference | -0.28018189639981955 | -0.28018189639981955 | -0.28018189639981955 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m4 | rer_at_50 | stats | RER ratio difference | -0.27499985926272147 | -0.27499985926272147 | -0.27499985926272147 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m4 | rer_at_50 | e1b | RER ratio difference | -0.22875811560834972 | -0.22875811560834972 | -0.22875811560834972 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m4 | rer_at_80 | msp | RER ratio difference | -0.21203063856576784 | -0.21203063856576784 | -0.21203063856576784 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m4 | rer_at_80 | stats | RER ratio difference | -0.2038501788372604 | -0.2038501788372604 | -0.2038501788372604 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m4 | rer_at_80 | e1b | RER ratio difference | -0.17697148665188842 | -0.17697148665188842 | -0.17697148665188842 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m8 | accuracy | msp | proportion difference | -0.1881371640407785 | -0.1881371640407785 | -0.1881371640407785 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m8 | auroc_correct | msp | AUROC difference | -0.03081971305497418 | -0.03081971305497418 | -0.03081971305497418 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m8 | auroc_correct | stats | AUROC difference | -0.02697575683060312 | -0.02697575683060312 | -0.02697575683060312 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m8 | auroc_correct | e1b | AUROC difference | 0.0057785183776157085 | 0.0057785183776157085 | 0.0057785183776157085 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m8 | e_aurc | msp | eAURC difference | 0.041432699786213736 | 0.041432699786213736 | 0.041432699786213736 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m8 | e_aurc | stats | eAURC difference | 0.03973453133672641 | 0.03973453133672641 | 0.03973453133672641 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m8 | e_aurc | e1b | eAURC difference | 0.026608119030351245 | 0.026608119030351245 | 0.026608119030351245 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m8 | rer_at_50 | msp | RER ratio difference | -0.24167534483211528 | -0.24167534483211528 | -0.24167534483211528 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m8 | rer_at_50 | stats | RER ratio difference | -0.23483018666368952 | -0.23483018666368952 | -0.23483018666368952 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m8 | rer_at_50 | e1b | RER ratio difference | -0.16110127539975416 | -0.16110127539975416 | -0.16110127539975416 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m8 | rer_at_80 | msp | RER ratio difference | -0.20362103351684172 | -0.20362103351684172 | -0.20362103351684172 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m8 | rer_at_80 | stats | RER ratio difference | -0.1978221564906606 | -0.1978221564906606 | -0.1978221564906606 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testA | RPN | phase1f_frozen_candidate_archive | testA | expb_m8 | rer_at_80 | e1b | RER ratio difference | -0.14948470978909 | -0.14948470978909 | -0.14948470978909 | 0.0 | 0.0 | 4316 | 644 | 1 | group_9fdabaa1eb0b61ca_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m2 | accuracy | msp | proportion difference | -0.16021260440394833 | -0.1601041327692808 | -0.16032107603861587 | -0.00010847163466753962 | 0.0 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m2 | auroc_correct | msp | AUROC difference | -0.07400981286743698 | -0.07397648687120924 | -0.07430049748300198 | -3.332599622773991e-05 | 0.0 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m2 | auroc_correct | stats | AUROC difference | -0.07488312598119162 | -0.07493533774402694 | -0.07513116182310602 | 5.221176283531914e-05 | 0.0 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m2 | auroc_correct | e1b | AUROC difference | -0.06104153218308269 | -0.06118086986286749 | -0.06115225965277268 | 0.00013933767978480116 | 0.0 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m2 | e_aurc | msp | eAURC difference | 0.07655339776931082 | 0.07653610428380218 | 0.07668656503134377 | 1.7293485508634854e-05 | 0.0 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m2 | e_aurc | stats | eAURC difference | 0.07617636035966235 | 0.0761884247808907 | 0.07627424465802894 | -1.2064421228354568e-05 | 4.6264939579029885e-18 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m2 | e_aurc | e1b | eAURC difference | 0.06797187658634009 | 0.06798677586120772 | 0.06797853950151457 | -1.4899274867622806e-05 | -4.6264939579029885e-18 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m2 | rer_at_50 | msp | RER ratio difference | -0.279149183711543 | -0.27801315841458224 | -0.2802924764370008 | -0.0011360252969608113 | 3.707971429900425e-17 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m2 | rer_at_50 | stats | RER ratio difference | -0.2840998871620238 | -0.2848472763772713 | -0.2839483556424825 | 0.0007473892152474922 | 3.69712940817557e-17 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m2 | rer_at_50 | e1b | RER ratio difference | -0.2589512724634165 | -0.2618361321138441 | -0.2596717662195906 | 0.0028848596504276367 | -1.8648277366750676e-17 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m2 | rer_at_80 | msp | RER ratio difference | -0.12368549640674242 | -0.1229341280533005 | -0.12403326203293145 | -0.0007513683534419215 | 4.662069341687669e-18 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m2 | rer_at_80 | stats | RER ratio difference | -0.12443190116880014 | -0.12328862475982645 | -0.12477932024229739 | -0.0011432764089736823 | -4.553649124439119e-18 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m2 | rer_at_80 | e1b | RER ratio difference | -0.11848970145049177 | -0.11748478243682943 | -0.11857027495132781 | -0.0010049190136623375 | -4.553649124439119e-18 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m4 | accuracy | msp | proportion difference | -0.19872003471092312 | -0.19861156307625558 | -0.19882850634559066 | -0.00010847163466753962 | 0.0 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m4 | auroc_correct | msp | AUROC difference | -0.0529347273341324 | -0.05270022385774468 | -0.05302423446953741 | -0.0002345034763877226 | -2.303929616531697e-18 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m4 | auroc_correct | stats | AUROC difference | -0.05256973746579056 | -0.052542533116200905 | -0.05273835719527997 | -2.7204349589656413e-05 | 4.62479989200848e-18 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m4 | auroc_correct | e1b | AUROC difference | -0.030572713702567755 | -0.03070124872343616 | -0.030672638513341344 | 0.0001285350208684033 | 0.0 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m4 | e_aurc | msp | eAURC difference | 0.07233250876157928 | 0.07219496815960519 | 0.07234542890714678 | 0.0001375406019740825 | 4.6349642873755315e-18 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m4 | e_aurc | stats | eAURC difference | 0.07057862982044773 | 0.0705803358995853 | 0.07066615577672353 | -1.7060791375663371e-06 | -9.251929124621909e-18 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m4 | e_aurc | e1b | eAURC difference | 0.058314105953352624 | 0.05838872750991738 | 0.058380491150224235 | -7.462155656475418e-05 | -2.3174821436877657e-18 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m4 | rer_at_50 | msp | RER ratio difference | -0.27222367354250193 | -0.27021590205797597 | -0.2724952200803945 | -0.002007771484525985 | 1.8648277366750676e-17 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m4 | rer_at_50 | stats | RER ratio difference | -0.2744314425938157 | -0.2768029872077049 | -0.27590406647291615 | 0.002371544613889239 | -1.8648277366750676e-17 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m4 | rer_at_50 | e1b | RER ratio difference | -0.23013581104657832 | -0.23298001831801526 | -0.23081565242376176 | 0.0028442072714369346 | 9.107298248878237e-18 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m4 | rer_at_80 | msp | RER ratio difference | -0.12485062863285606 | -0.12382554220213608 | -0.12492467618176702 | -0.0010250864307199865 | 4.553649124439119e-18 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m4 | rer_at_80 | stats | RER ratio difference | -0.1272097177106518 | -0.12579230797103305 | -0.127283003453504 | -0.0014174097396187357 | 0.0 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m4 | rer_at_80 | e1b | RER ratio difference | -0.11492989030546792 | -0.11366887712299911 | -0.11475436963749747 | -0.0012610131824688324 | 2.3201926491189795e-17 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m8 | accuracy | msp | proportion difference | -0.21412300683371296 | -0.2142314784683805 | -0.21444842173771558 | 0.00010847163466753962 | 0.0 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m8 | auroc_correct | msp | AUROC difference | -0.022044473076632043 | -0.022119964744262033 | -0.022443975356054763 | 7.549166762998993e-05 | 0.0 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m8 | auroc_correct | stats | AUROC difference | -0.02252407558678852 | -0.022706449463622397 | -0.022902273542701468 | 0.0001823738768338782 | -2.303929616531697e-18 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m8 | auroc_correct | e1b | AUROC difference | 0.011420902317484235 | 0.01105204108525478 | 0.011080651295349594 | 0.000368861232229456 | 0.0 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m8 | e_aurc | msp | eAURC difference | 0.054340290378587015 | 0.0546760745654238 | 0.05482653531296539 | -0.0003357841868367832 | -4.607859233063394e-18 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m8 | e_aurc | stats | eAURC difference | 0.05291487229123041 | 0.05325621228361865 | 0.05334203216075687 | -0.0003413399923882325 | -4.607859233063394e-18 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m8 | e_aurc | e1b | eAURC difference | 0.03540842921209303 | 0.03580650560898065 | 0.03579826924928751 | -0.0003980763968876218 | -2.3310346708438345e-18 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m8 | rer_at_50 | msp | RER ratio difference | -0.2373335983879338 | -0.2358528702111882 | -0.23813218823360674 | -0.0014807281767456233 | 1.8431436932253575e-17 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m8 | rer_at_50 | stats | RER ratio difference | -0.243373853228999 | -0.24506820289176767 | -0.2441692821569789 | 0.0016943496627686756 | 9.324138683375338e-18 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m8 | rer_at_50 | e1b | RER ratio difference | -0.1875243311089586 | -0.19013055627830747 | -0.18796619038405402 | 0.002606225169348886 | -2.7755575615628914e-17 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m8 | rer_at_80 | msp | RER ratio difference | -0.1197442180871275 | -0.11885940847225768 | -0.11995854245188864 | -0.0008848096148698312 | 9.215718466126788e-18 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m8 | rer_at_80 | stats | RER ratio difference | -0.12114302792765141 | -0.11986722554767619 | -0.12135792103014713 | -0.0012758023799752244 | 0.0 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |
| RPN__phase1f_frozen_candidate_archive__dose_joint__testB | RPN | phase1f_frozen_candidate_archive | testB | expb_m8 | rer_at_80 | e1b | RER ratio difference | -0.09782730115154416 | -0.09697164297632148 | -0.09805713549081983 | -0.0008556581752226974 | 1.8539857149502126e-17 | 3073 | 544 | 1 | group_10881c9667323aeb_raw.npz |

## audit_run_20261003_Cslot_audit4/forward/pilot_sensitivity_one_rep_20261003_01/summary.json

Source: `results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/forward/pilot_sensitivity_one_rep_20261003_01/summary.json`; SHA256 `79eb64f86a71ce221eb6c6614ce1e40b258e487568a6e26b8f4459ae5e20230b`.

| Field | Stored value |
|---|---|
| schema | research-repair-v1-candidate-sensitivity-summary-v1 |
| status | PILOT_PARTIAL |
| formal | False |
| audit_run_id | 20261003_Cslot_audit4 |
| candidate_audit_summary_sha256 | d1a7eecac75467a7a46bae3133448451c57aa7a0c8550a4aca1da3cf3fe6cce8 |
| candidate_forward_summary_sha256 | ab93b666d64be0925fae8acc09fab2c7b308fc44eb06cc1b52b271c6f01a7250 |
| predictions_manifest.path | results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/forward/predictions_manifest.json |
| predictions_manifest.sha256 | 315fd7aa9263fb901b706e352efc18cb9e5cece60de0ff4d06ac1be7f5e223ba |
| bootstrap_contract.resample_unit | image_cluster |
| bootstrap_contract.n_replicates | 1 |
| bootstrap_contract.seed | 0 |
| bootstrap_contract.ci_level | 0.95 |
| bootstrap_contract.method | percentile |
| bootstrap_contract.estimand_orientation | old minus corrected candidate; fractions and RER ratio remain unscaled |
| bootstrap_contract.no_percentage_share_claims | True |
| bootstrap.resample_unit | image_cluster |
| bootstrap.n_replicates | 1 |
| bootstrap.seed | 0 |
| bootstrap.ci_level | 0.95 |
| bootstrap.method | percentile |
| dose_estimand_contract.old_arm | metric(old_m) - metric(old_m0) |
| dose_estimand_contract.corrected_arm | metric(corrected_m) - metric(corrected_m0) |
| dose_estimand_contract.version_change | [metric(old_m) - metric(corrected_m)] - [metric(old_m0) - metric(corrected_m0)] |
| dose_estimand_contract.fixed_original_m0_contrast | metric(corrected_m) - metric(original old_m0); descriptive only, not corrected baseline |
| nonforwardable_empty_source_cohorts | [] |
| estimates_csv | estimates.csv |
| n_estimates | 1404 |
| generated_utc | 2026-10-03T08:35:28Z |

## audit_run_20261003_Cslot_audit4/forward/sensitivity_bootstrap/summary.json

Source: `results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/forward/sensitivity_bootstrap/summary.json`; SHA256 `c471927ba20f44bc7ee65ee480ac9954a2e027ea7b1f47cfe05938e37464b550`.

| Field | Stored value |
|---|---|
| schema | research-repair-v1-candidate-sensitivity-summary-v1 |
| status | COMPLETE |
| formal | True |
| audit_run_id | 20261003_Cslot_audit4 |
| candidate_audit_summary_sha256 | d1a7eecac75467a7a46bae3133448451c57aa7a0c8550a4aca1da3cf3fe6cce8 |
| candidate_forward_summary_sha256 | ab93b666d64be0925fae8acc09fab2c7b308fc44eb06cc1b52b271c6f01a7250 |
| predictions_manifest.path | results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/forward/predictions_manifest.json |
| predictions_manifest.sha256 | 315fd7aa9263fb901b706e352efc18cb9e5cece60de0ff4d06ac1be7f5e223ba |
| bootstrap_contract.resample_unit | image_cluster |
| bootstrap_contract.n_replicates | 5000 |
| bootstrap_contract.seed | 0 |
| bootstrap_contract.ci_level | 0.95 |
| bootstrap_contract.method | percentile |
| bootstrap_contract.estimand_orientation | old minus corrected candidate; fractions and RER ratio remain unscaled |
| bootstrap_contract.no_percentage_share_claims | True |
| bootstrap.resample_unit | image_cluster |
| bootstrap.n_replicates | 5000 |
| bootstrap.seed | 0 |
| bootstrap.ci_level | 0.95 |
| bootstrap.method | percentile |
| dose_estimand_contract.old_arm | metric(old_m) - metric(old_m0) |
| dose_estimand_contract.corrected_arm | metric(corrected_m) - metric(corrected_m0) |
| dose_estimand_contract.version_change | [metric(old_m) - metric(corrected_m)] - [metric(old_m0) - metric(corrected_m0)] |
| dose_estimand_contract.fixed_original_m0_contrast | metric(corrected_m) - metric(original old_m0); descriptive only, not corrected baseline |
| nonforwardable_empty_source_cohorts | [] |
| estimates_csv | estimates.csv |
| n_estimates | 1404 |
| generated_utc | 2026-10-03T13:12:48Z |

## audit_run_20261003_Cslot_audit4/forward/summary.json

Source: `results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/forward/summary.json`; SHA256 `ab93b666d64be0925fae8acc09fab2c7b308fc44eb06cc1b52b271c6f01a7250`.

| Field | Stored value |
|---|---|
| schema | research-repair-v1-candidate-forward-summary-v1 |
| status | COMPLETE |
| preflight_only | False |
| preflight_rows_per_source | None |
| audit_run_id | 20261003_Cslot_audit4 |
| candidate_audit_pointer.path | results/research_repair_v1/candidates/current_audit.json |
| candidate_audit_pointer.summary_path | results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/summary.json |
| candidate_audit_pointer.summary_sha256 | d1a7eecac75467a7a46bae3133448451c57aa7a0c8550a4aca1da3cf3fe6cce8 |
| candidate_audit_pointer.candidate_indices_path | results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/candidate_sets_true_target_category_v1.npz |
| candidate_audit_pointer.candidate_indices_sha256 | 4f93780ddff4fb26fc1c8e3894ffba3dd172d96068394e8fb37672ede9f26b2d |
| method | frozen B3 scorer plus frozen load-and-predict Stats/E1b bundle; no refit, training, extraction, or download |
| anchor_contract.old_candidate_first | True |
| anchor_contract.candidate_id_source | hash-pinned versioned candidate NPZ; paired source cohorts are checked by canonical sentence_id/ref_id/image_id |
| anchor_contract.raw_score_tolerance | 0.0001 |
| anchor_contract.reliability_tolerance | 1e-09 |
| anchor_contract.reliability_bundle_verification.schema_version | a11-frozen-reliability-v1 |
| anchor_contract.reliability_bundle_verification.tolerance | 1e-09 |
| anchor_contract.reliability_bundle_verification.scorers | ["b3_seed1", "b3_seed2", "b3_seed3"] |
| anchor_contract.reliability_bundle_verification.checks.b3_seed1.e1b_coefficients_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed1.stats17_normalisation_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed1.temperature_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed1.stats_logistic_pred_max_abs | 2.220446049250313e-16 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed2.e1b_coefficients_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed2.stats17_normalisation_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed2.temperature_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed2.stats_logistic_pred_max_abs | 2.220446049250313e-16 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed3.e1b_coefficients_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed3.stats17_normalisation_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed3.temperature_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed3.stats_logistic_pred_max_abs | 2.220446049250313e-16 |
| anchor_contract.reliability_bundle_verification.all.e1b_coefficients_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.all.stats17_normalisation_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.all.stats_logistic_pred_max_abs | 2.220446049250313e-16 |
| anchor_contract.reliability_bundle_verification.all.temperature_max_abs | 0.0 |
| cohort_contract.source_cohorts_remain_separate | True |
| cohort_contract.intersection | historical old candidate availability ∩ corrected true-target-category availability, per family/source/cell |
| cohort_contract.row_order | ascending sentence_id; no ref_id joins or deduplication |
| predictions_manifest.path | results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/forward/predictions_manifest.json |
| predictions_manifest.size_bytes | 68075 |
| predictions_manifest.sha256 | 315fd7aa9263fb901b706e352efc18cb9e5cece60de0ff4d06ac1be7f5e223ba |
| predictions_manifest.schema | research-repair-v1-candidate-forward-manifest-v1 |
| bootstrap.status | NOT_RUN_BY_C_FORWARD |
| bootstrap.consumer | C candidate sensitivity runner using ccg.repairs.statistics.joint_prediction_bootstrap |
| bootstrap.planned_output | results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/forward/sensitivity_bootstrap |
| inputs.baseline_manifest_path | results/research_repair_v1/input_manifest.json |
| inputs.baseline_manifest_sha256 | 511499c47659bde0bdf207befac4e6efb41975a8b4f1623b10e469c0b89a334f |
| inputs.baseline_covered_input_count | 61 |
| inputs.baseline_covered_inputs.cache/image_sizes.npz.path | cache/image_sizes.npz |
| inputs.baseline_covered_inputs.cache/image_sizes.npz.size_bytes | 5190224 |
| inputs.baseline_covered_inputs.cache/image_sizes.npz.sha256 | d8c9605f9f32c320aec18809f1ac05200b3cbf1177e417f1aa3d243e4f1dfdc8 |
| inputs.baseline_covered_inputs.cache/proposals.h5.path | cache/proposals.h5 |
| inputs.baseline_covered_inputs.cache/proposals.h5.size_bytes | 60949136 |
| inputs.baseline_covered_inputs.cache/proposals.h5.sha256 | de0c769fa4266d1b9769abe6c8c7587dd8d221ad45a465503b17974fe1057ad6 |
| inputs.baseline_covered_inputs.cache/proposals_detr_r50.h5.path | cache/proposals_detr_r50.h5 |
| inputs.baseline_covered_inputs.cache/proposals_detr_r50.h5.size_bytes | 60949136 |
| inputs.baseline_covered_inputs.cache/proposals_detr_r50.h5.sha256 | 47a63ec2e6e1a7fe12841b6f5bdf702712beaecf5f69ab0cf661dc904da9ec18 |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/models.json.path | results/phase1e_refcocog_external/frozen_models/models.json |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/models.json.size_bytes | 17478 |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/models.json.sha256 | 689173894f6c280ed354bc64688c9819cacebce3fba3835cdbff59313c07c241 |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/models.sha256.path | results/phase1e_refcocog_external/frozen_models/models.sha256 |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/models.sha256.size_bytes | 230 |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/models.sha256.sha256 | 2111164a02f3b8f390ac30ca4db07dafbef46a701f66bb210bd19a12e76bdac1 |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/frozen_artifact_manifest.json.path | results/phase1e_refcocog_external/frozen_models/frozen_artifact_manifest.json |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/frozen_artifact_manifest.json.size_bytes | 16505 |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/frozen_artifact_manifest.json.sha256 | 9bed2cc73b0f0a9ad8c6ccd014c264c0a47fc56ad9106537a2e21237451e4eb6 |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/bundle_verification.json.path | results/phase1e_refcocog_external/frozen_models/bundle_verification.json |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/bundle_verification.json.size_bytes | 981 |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/bundle_verification.json.sha256 | 163e33ac8ef20be4c18d517cfc436c71f4bae2775f44d307cd4ab26dc476d286 |
| inputs.baseline_covered_inputs.cache/features/metadata.json.path | cache/features/metadata.json |
| inputs.baseline_covered_inputs.cache/features/metadata.json.size_bytes | 3263 |
| inputs.baseline_covered_inputs.cache/features/metadata.json.sha256 | c8cb592ae28ee5c498218a2b0a3d7e02de1472e3ab858040e397a0508e59883b |
| inputs.baseline_covered_inputs.cache/features/region_features.h5.path | cache/features/region_features.h5 |
| inputs.baseline_covered_inputs.cache/features/region_features.h5.size_bytes | 1320262656 |
| inputs.baseline_covered_inputs.cache/features/region_features.h5.sha256 | a97fd40946e79b0f79373fb7228cea85868f18a101d6360fe027742b3f4d7481 |
| inputs.baseline_covered_inputs.cache/features/text_features.h5.path | cache/features/text_features.h5 |
| inputs.baseline_covered_inputs.cache/features/text_features.h5.size_bytes | 154883968 |
| inputs.baseline_covered_inputs.cache/features/text_features.h5.sha256 | 82c0d10195121bcefbf11e2adfab03348377e5aae6a7288e76568b8da25617a1 |
| inputs.baseline_covered_inputs.cache/features/text_index.csv.path | cache/features/text_index.csv |
| inputs.baseline_covered_inputs.cache/features/text_index.csv.size_bytes | 7258137 |
| inputs.baseline_covered_inputs.cache/features/text_index.csv.sha256 | bc4248b529d9d9c6241b16bc871fe4ce1d43fab6c22213501d852989acaf43a1 |
| inputs.baseline_covered_inputs.cache/manifests/random_testA.jsonl.path | cache/manifests/random_testA.jsonl |
| inputs.baseline_covered_inputs.cache/manifests/random_testA.jsonl.size_bytes | 1252029 |
| inputs.baseline_covered_inputs.cache/manifests/random_testA.jsonl.sha256 | 35bfd08298df560c7042ec8ebac3c6e6daf115b410d4cffc100d24dee937a1dd |
| inputs.baseline_covered_inputs.cache/manifests/random_testB.jsonl.path | cache/manifests/random_testB.jsonl |
| inputs.baseline_covered_inputs.cache/manifests/random_testB.jsonl.size_bytes | 1137120 |
| inputs.baseline_covered_inputs.cache/manifests/random_testB.jsonl.sha256 | 32de6c535df1b9679a931e131baef74483cdd06cdf959c23cc0d13a394782259 |
| inputs.baseline_covered_inputs.cache/manifests/same_category_testA.jsonl.path | cache/manifests/same_category_testA.jsonl |
| inputs.baseline_covered_inputs.cache/manifests/same_category_testA.jsonl.size_bytes | 1313457 |
| inputs.baseline_covered_inputs.cache/manifests/same_category_testA.jsonl.sha256 | c5425b202ddb9a82db0715255a10e2c6b59e6dd66662dfd1c3c8caebcf67a8ee |
| inputs.baseline_covered_inputs.cache/manifests/same_category_testB.jsonl.path | cache/manifests/same_category_testB.jsonl |
| inputs.baseline_covered_inputs.cache/manifests/same_category_testB.jsonl.size_bytes | 1196525 |
| inputs.baseline_covered_inputs.cache/manifests/same_category_testB.jsonl.sha256 | 876d671dab5174532ff5799506bfffe32fdee95ac63d1bfdafcfd6616e2ead5e |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/random_testA.jsonl.path | cache/manifests_detr/manifests/random_testA.jsonl |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/random_testA.jsonl.size_bytes | 1267143 |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/random_testA.jsonl.sha256 | 176837d2ba53712e95e8edf9ba71a98a456d0d9814d5b682fd5b924f35fd388c |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/random_testB.jsonl.path | cache/manifests_detr/manifests/random_testB.jsonl |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/random_testB.jsonl.size_bytes | 1127679 |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/random_testB.jsonl.sha256 | 641b30aba5405ec657e6a99f92034706f4f3b00ef36c453ca0e132eec281ab70 |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/same_category_testA.jsonl.path | cache/manifests_detr/manifests/same_category_testA.jsonl |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/same_category_testA.jsonl.size_bytes | 1343514 |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/same_category_testA.jsonl.sha256 | 604a73f38248561c41b30864286395347ed41fbbc0e31f1f0a306c591f85727b |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/same_category_testB.jsonl.path | cache/manifests_detr/manifests/same_category_testB.jsonl |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/same_category_testB.jsonl.size_bytes | 1187573 |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/same_category_testB.jsonl.sha256 | fcb3a2ff43b6b8a23e8fccd2e05aad4605bd507cfbc692a7a3fb94d26272bc4c |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_1/model.npz.path | results/phase0b_independent/seed_1/model.npz |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_1/model.npz.size_bytes | 858590 |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_1/model.npz.sha256 | 396d90da2f4f402e0dba86f4956c12efdf19b6b0c76a542e5b60a281810a8e74 |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_1/training.json.path | results/phase0b_independent/seed_1/training.json |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_1/training.json.size_bytes | 2452 |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_1/training.json.sha256 | b680a6784092d10c9a7408bffcba57a1b7cf498b41bac8b341fc808914598ea1 |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_2/model.npz.path | results/phase0b_independent/seed_2/model.npz |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_2/model.npz.size_bytes | 858590 |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_2/model.npz.sha256 | 6092592e6eef3cc031f9f3c0a33fcd296ea31061be80ab5008ceb8c6d7c910c4 |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_2/training.json.path | results/phase0b_independent/seed_2/training.json |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_2/training.json.size_bytes | 2558 |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_2/training.json.sha256 | 9dc0bab998b71aab3cc2c969217acf7d6ab218dc401a4edfd7db493b943408d2 |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_3/model.npz.path | results/phase0b_independent/seed_3/model.npz |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_3/model.npz.size_bytes | 858590 |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_3/model.npz.sha256 | b7671881b4c441d0ab01da1ee91572d2c3d66067c2a131e29e700460217d14ab |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_3/training.json.path | results/phase0b_independent/seed_3/training.json |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_3/training.json.size_bytes | 2347 |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_3/training.json.sha256 | 7ea10c653636d4645ec9ad6ab9126940bc38cf1ae0a64ee6d9d1453240a7e43f |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard5__b3_seed1.npz.path | results/phase1f_hard_semantic/predictions/hard5__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard5__b3_seed1.npz.size_bytes | 488639 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard5__b3_seed1.npz.sha256 | eeb01bccd298c5bc1399b5ad47c2d20860c0832a1bd3c86c197bd6411b02bb67 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand5__b3_seed1.npz.path | results/phase1f_hard_semantic/predictions/rand5__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand5__b3_seed1.npz.size_bytes | 492564 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand5__b3_seed1.npz.sha256 | 535ce06d00727ddf75d85ac9d8168023495733c9df67f9d01c24ae03c28cd189 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard5__b3_seed2.npz.path | results/phase1f_hard_semantic/predictions/hard5__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard5__b3_seed2.npz.size_bytes | 490808 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard5__b3_seed2.npz.sha256 | c04ed006bd622baff749929abe971eb6e48801c118f20df72065c89779fa81c8 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand5__b3_seed2.npz.path | results/phase1f_hard_semantic/predictions/rand5__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand5__b3_seed2.npz.size_bytes | 495849 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand5__b3_seed2.npz.sha256 | 553ae678e3f25a39ab3b5dfebe5ccca8af1f977e6338128fa567d63af86f93bc |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard5__b3_seed3.npz.path | results/phase1f_hard_semantic/predictions/hard5__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard5__b3_seed3.npz.size_bytes | 491673 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard5__b3_seed3.npz.sha256 | df1b2366f6540ff2b7dd8cb0cf49a44c47ee07b6309a9d65b299a2cd4e1c1df0 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand5__b3_seed3.npz.path | results/phase1f_hard_semantic/predictions/rand5__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand5__b3_seed3.npz.size_bytes | 496626 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand5__b3_seed3.npz.sha256 | 24965e381f0d58b34ec329f33deff2b626a1dbff7c40324a7505ff62ce0117e1 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard10__b3_seed1.npz.path | results/phase1f_hard_semantic/predictions/hard10__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard10__b3_seed1.npz.size_bytes | 461545 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard10__b3_seed1.npz.sha256 | 19d56d015396caadb15a7c62737b0af3bd93ec5bd428948bc2204db8d50f4f91 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand10__b3_seed1.npz.path | results/phase1f_hard_semantic/predictions/rand10__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand10__b3_seed1.npz.size_bytes | 466178 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand10__b3_seed1.npz.sha256 | 3e862e196843aae58a7ba90328e863fe55ce86d84719dba33392d962233eaeb6 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard10__b3_seed2.npz.path | results/phase1f_hard_semantic/predictions/hard10__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard10__b3_seed2.npz.size_bytes | 464545 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard10__b3_seed2.npz.sha256 | 05922bc114132d58dbb5c71bea10b8a74a6df92948a1a3ce8d935a79ab287c9c |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand10__b3_seed2.npz.path | results/phase1f_hard_semantic/predictions/rand10__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand10__b3_seed2.npz.size_bytes | 470154 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand10__b3_seed2.npz.sha256 | 5a69d81380016012ceb44cb3ee2b2a98340156ae7205153014afd304faa90454 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard10__b3_seed3.npz.path | results/phase1f_hard_semantic/predictions/hard10__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard10__b3_seed3.npz.size_bytes | 465756 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard10__b3_seed3.npz.sha256 | 767eeab700119876be39ee8a77a1298af6499e6a819c84aec2b2c825f6f3682d |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand10__b3_seed3.npz.path | results/phase1f_hard_semantic/predictions/rand10__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand10__b3_seed3.npz.size_bytes | 471045 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand10__b3_seed3.npz.sha256 | a8736cd10b1734c5a2087a439371a0e54a41b31578c5e895621e88b16c778caa |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m0__b3_seed1.npz.path | results/phase1f_hard_semantic/predictions/expb_m0__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m0__b3_seed1.npz.size_bytes | 512453 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m0__b3_seed1.npz.sha256 | 585d51d906282c6b4f816357953f59908c592cb6c633d9f296055093f5b05ef0 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m0__b3_seed2.npz.path | results/phase1f_hard_semantic/predictions/expb_m0__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m0__b3_seed2.npz.size_bytes | 516573 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m0__b3_seed2.npz.sha256 | d590e650744e1e24c6743ae2f85b57d5abdde313d1d07a3c4d05b1dbee080fc1 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m0__b3_seed3.npz.path | results/phase1f_hard_semantic/predictions/expb_m0__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m0__b3_seed3.npz.size_bytes | 517703 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m0__b3_seed3.npz.sha256 | bd1c11ce9ec056f0b99a54ee98c0d05a68452688df5c22612efc5c0e388f42bb |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m2__b3_seed1.npz.path | results/phase1f_hard_semantic/predictions/expb_m2__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m2__b3_seed1.npz.size_bytes | 510602 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m2__b3_seed1.npz.sha256 | f6159962c4a22396dcfcdeec13b4f21b7a6b9e44cf9227fd2437e17aaaf23169 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m2__b3_seed2.npz.path | results/phase1f_hard_semantic/predictions/expb_m2__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m2__b3_seed2.npz.size_bytes | 514551 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m2__b3_seed2.npz.sha256 | 3311faeb59a840a0b9b18c4e199431fdb677910902185bd1e088f4b6de843f13 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m2__b3_seed3.npz.path | results/phase1f_hard_semantic/predictions/expb_m2__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m2__b3_seed3.npz.size_bytes | 515814 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m2__b3_seed3.npz.sha256 | 97bb7545b87012633393318984442a9ea09ebe023739838a278c08180b9fc93c |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m4__b3_seed1.npz.path | results/phase1f_hard_semantic/predictions/expb_m4__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m4__b3_seed1.npz.size_bytes | 508989 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m4__b3_seed1.npz.sha256 | c389d470983ed8abe9fdfd68a74fb09e4bbb425a67516cd404ff44c2f7b8866c |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m4__b3_seed2.npz.path | results/phase1f_hard_semantic/predictions/expb_m4__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m4__b3_seed2.npz.size_bytes | 512831 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m4__b3_seed2.npz.sha256 | 4500da5acb71103c0f3e87e013fee6f4e30fbce1ffa934a5e725bd54d95aeb4b |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m4__b3_seed3.npz.path | results/phase1f_hard_semantic/predictions/expb_m4__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m4__b3_seed3.npz.size_bytes | 514200 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m4__b3_seed3.npz.sha256 | 95946dd44fdb53d768c99a7f8ba4bed00c48bc1dcfeb6114a5d465a10e40d2e1 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m8__b3_seed1.npz.path | results/phase1f_hard_semantic/predictions/expb_m8__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m8__b3_seed1.npz.size_bytes | 506095 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m8__b3_seed1.npz.sha256 | ce43d52108ed5c882bb0ede61fc6d519f9aded737c92103f3ab5150ed6a7ca21 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m8__b3_seed2.npz.path | results/phase1f_hard_semantic/predictions/expb_m8__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m8__b3_seed2.npz.size_bytes | 509544 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m8__b3_seed2.npz.sha256 | b534fec8797ecd03ee9e004b6116eef9466ad0a18e326c88932f16b0cf97bdd2 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m8__b3_seed3.npz.path | results/phase1f_hard_semantic/predictions/expb_m8__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m8__b3_seed3.npz.size_bytes | 510923 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m8__b3_seed3.npz.sha256 | b325c0236226b917ea0980c998d5c8ff83e8490f823943ce9aaeb8f0eb8bcefb |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed1.npz.path | results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed1.npz.size_bytes | 1773140 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed1.npz.sha256 | 048b2c5bbc620488496a513526e770bf2820776dfd39f2ef88d01969b36f657d |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed1.npz.path | results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed1.npz.size_bytes | 1920708 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed1.npz.sha256 | 0f721d6656bdb24fa464194dc144cf24394b32aab7515cd7356a9508070bcbbe |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed2.npz.path | results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed2.npz.size_bytes | 1773140 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed2.npz.sha256 | 1acaf8454d8b41edde7784d272234e7d77bbdcb48423ea70b8e8a9c947917ae5 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed2.npz.path | results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed2.npz.size_bytes | 1920708 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed2.npz.sha256 | bc3b78e7e6df70a878fd1b7da2a5ba0ec6f6f974c84e4c6c1e77884d7e59e847 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed3.npz.path | results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed3.npz.size_bytes | 1773140 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed3.npz.sha256 | 0555bd968db47cb0a048305bb5a1ccf384e9d56943422fd816386a9e7965e516 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed3.npz.path | results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed3.npz.size_bytes | 1920708 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed3.npz.sha256 | b70f7b6ce9e9de227925422dcaeb42b7c1f157eada1c53246940d118f38cd751 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed1.npz.path | results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed1.npz.size_bytes | 1543692 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed1.npz.sha256 | 392453e8d549d917f765c169e9979c923178a1eac309cb26a6b5736af8b69cab |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed1.npz.path | results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed1.npz.size_bytes | 1953092 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed1.npz.sha256 | 335c34a29a9450f2f75d9c427fce1f5105ffc1c937445f55a7892f0926b5a523 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed2.npz.path | results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed2.npz.size_bytes | 1543692 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed2.npz.sha256 | a3ff9b73fcd566ec80c582d9fb3fa6a914c4a612d4529a5abbe5b4dc2e1d24de |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed2.npz.path | results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed2.npz.size_bytes | 1953092 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed2.npz.sha256 | f0e1391b57f1943ce606ff0ae128e3c897d34deba883037bc995f9257a907f12 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed3.npz.path | results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed3.npz.size_bytes | 1543692 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed3.npz.sha256 | 9d20f4fba2ddf600c71fd1d492e279703291c52b9c1675a19fdcddbae275c300 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed3.npz.path | results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed3.npz.size_bytes | 1953092 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed3.npz.sha256 | 8464221ea8360e1c37b3a0a4bc144ff9142a8d14662266f7a039eaad3e0a5fd8 |
| inputs.candidate_coco_supplemental_manifest | results/research_repair_v1/candidates/supplemental_input_manifest.json |
| inputs.candidate_cooco_sha256 | 777ef504abfeba349930485bb32ecd5b6529129a3e010f7852e1437cb049b732 |
| inputs.detr_features_supplemental_manifest | results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/forward_preflight_rows2_20261003T155906_55728/supplemental_input_manifest.json |
| inputs.detr_features_manifest_sha256 | 7b57d233258f742aae296f67514275c5a4750b9e7923240c9dbb2f9421bd54df |
| device | cuda |
| device_name | NVIDIA GeForce RTX 4060 Laptop GPU |
| forward_seconds | 447.508 |
| new_training | 0 |
| new_feature_extraction | 0 |
| model_refits | 0 |
| generated_utc | 2026-10-03T08:27:25Z |
| peak_vram_bytes | 16746496 |

## audit_run_20261003_Cslot_audit4/forward_preflight_rows2_20261003T161731_54376/summary.json

Source: `results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/forward_preflight_rows2_20261003T161731_54376/summary.json`; SHA256 `b05fe9d5dc619aa7f7038eb97988a8604b3a6d4bd94e633af858b3e1a08f825d`.

| Field | Stored value |
|---|---|
| schema | research-repair-v1-candidate-forward-preflight-v1 |
| status | PREFLIGHT_PASS |
| preflight_only | True |
| preflight_rows_per_source | 2 |
| audit_run_id | 20261003_Cslot_audit4 |
| candidate_audit_pointer.path | results/research_repair_v1/candidates/current_audit.json |
| candidate_audit_pointer.summary_path | results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/summary.json |
| candidate_audit_pointer.summary_sha256 | d1a7eecac75467a7a46bae3133448451c57aa7a0c8550a4aca1da3cf3fe6cce8 |
| candidate_audit_pointer.candidate_indices_path | results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/candidate_sets_true_target_category_v1.npz |
| candidate_audit_pointer.candidate_indices_sha256 | 4f93780ddff4fb26fc1c8e3894ffba3dd172d96068394e8fb37672ede9f26b2d |
| method | frozen B3 scorer plus frozen load-and-predict Stats/E1b bundle; no refit, training, extraction, or download |
| anchor_contract.old_candidate_first | True |
| anchor_contract.candidate_id_source | hash-pinned versioned candidate NPZ; paired source cohorts are checked by canonical sentence_id/ref_id/image_id |
| anchor_contract.raw_score_tolerance | 0.0001 |
| anchor_contract.reliability_tolerance | 1e-09 |
| anchor_contract.reliability_bundle_verification.schema_version | a11-frozen-reliability-v1 |
| anchor_contract.reliability_bundle_verification.tolerance | 1e-09 |
| anchor_contract.reliability_bundle_verification.scorers | ["b3_seed1", "b3_seed2", "b3_seed3"] |
| anchor_contract.reliability_bundle_verification.checks.b3_seed1.e1b_coefficients_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed1.stats17_normalisation_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed1.temperature_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed1.stats_logistic_pred_max_abs | 2.220446049250313e-16 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed2.e1b_coefficients_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed2.stats17_normalisation_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed2.temperature_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed2.stats_logistic_pred_max_abs | 2.220446049250313e-16 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed3.e1b_coefficients_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed3.stats17_normalisation_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed3.temperature_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.checks.b3_seed3.stats_logistic_pred_max_abs | 2.220446049250313e-16 |
| anchor_contract.reliability_bundle_verification.all.e1b_coefficients_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.all.stats17_normalisation_max_abs | 0.0 |
| anchor_contract.reliability_bundle_verification.all.stats_logistic_pred_max_abs | 2.220446049250313e-16 |
| anchor_contract.reliability_bundle_verification.all.temperature_max_abs | 0.0 |
| cohort_contract.source_cohorts_remain_separate | True |
| cohort_contract.intersection | historical old candidate availability ∩ corrected true-target-category availability, per family/source/cell |
| cohort_contract.row_order | ascending sentence_id; no ref_id joins or deduplication |
| predictions_manifest.path | results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/forward_preflight_rows2_20261003T161731_54376/predictions_manifest.json |
| predictions_manifest.size_bytes | 67837 |
| predictions_manifest.sha256 | f81dc5e84344d0a64856463fed19b15a4d47c4c24746bc7b4556a2d03dea721c |
| predictions_manifest.schema | research-repair-v1-candidate-forward-preflight-manifest-v1 |
| bootstrap.status | NOT_RUN_BY_C_FORWARD |
| bootstrap.consumer | C candidate sensitivity runner using ccg.repairs.statistics.joint_prediction_bootstrap |
| bootstrap.planned_output | results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/forward_preflight_rows2_20261003T161731_54376/sensitivity_bootstrap |
| inputs.baseline_manifest_path | results/research_repair_v1/input_manifest.json |
| inputs.baseline_manifest_sha256 | 511499c47659bde0bdf207befac4e6efb41975a8b4f1623b10e469c0b89a334f |
| inputs.baseline_covered_input_count | 61 |
| inputs.baseline_covered_inputs.cache/image_sizes.npz.path | cache/image_sizes.npz |
| inputs.baseline_covered_inputs.cache/image_sizes.npz.size_bytes | 5190224 |
| inputs.baseline_covered_inputs.cache/image_sizes.npz.sha256 | d8c9605f9f32c320aec18809f1ac05200b3cbf1177e417f1aa3d243e4f1dfdc8 |
| inputs.baseline_covered_inputs.cache/proposals.h5.path | cache/proposals.h5 |
| inputs.baseline_covered_inputs.cache/proposals.h5.size_bytes | 60949136 |
| inputs.baseline_covered_inputs.cache/proposals.h5.sha256 | de0c769fa4266d1b9769abe6c8c7587dd8d221ad45a465503b17974fe1057ad6 |
| inputs.baseline_covered_inputs.cache/proposals_detr_r50.h5.path | cache/proposals_detr_r50.h5 |
| inputs.baseline_covered_inputs.cache/proposals_detr_r50.h5.size_bytes | 60949136 |
| inputs.baseline_covered_inputs.cache/proposals_detr_r50.h5.sha256 | 47a63ec2e6e1a7fe12841b6f5bdf702712beaecf5f69ab0cf661dc904da9ec18 |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/models.json.path | results/phase1e_refcocog_external/frozen_models/models.json |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/models.json.size_bytes | 17478 |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/models.json.sha256 | 689173894f6c280ed354bc64688c9819cacebce3fba3835cdbff59313c07c241 |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/models.sha256.path | results/phase1e_refcocog_external/frozen_models/models.sha256 |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/models.sha256.size_bytes | 230 |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/models.sha256.sha256 | 2111164a02f3b8f390ac30ca4db07dafbef46a701f66bb210bd19a12e76bdac1 |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/frozen_artifact_manifest.json.path | results/phase1e_refcocog_external/frozen_models/frozen_artifact_manifest.json |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/frozen_artifact_manifest.json.size_bytes | 16505 |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/frozen_artifact_manifest.json.sha256 | 9bed2cc73b0f0a9ad8c6ccd014c264c0a47fc56ad9106537a2e21237451e4eb6 |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/bundle_verification.json.path | results/phase1e_refcocog_external/frozen_models/bundle_verification.json |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/bundle_verification.json.size_bytes | 981 |
| inputs.baseline_covered_inputs.results/phase1e_refcocog_external/frozen_models/bundle_verification.json.sha256 | 163e33ac8ef20be4c18d517cfc436c71f4bae2775f44d307cd4ab26dc476d286 |
| inputs.baseline_covered_inputs.cache/features/metadata.json.path | cache/features/metadata.json |
| inputs.baseline_covered_inputs.cache/features/metadata.json.size_bytes | 3263 |
| inputs.baseline_covered_inputs.cache/features/metadata.json.sha256 | c8cb592ae28ee5c498218a2b0a3d7e02de1472e3ab858040e397a0508e59883b |
| inputs.baseline_covered_inputs.cache/features/region_features.h5.path | cache/features/region_features.h5 |
| inputs.baseline_covered_inputs.cache/features/region_features.h5.size_bytes | 1320262656 |
| inputs.baseline_covered_inputs.cache/features/region_features.h5.sha256 | a97fd40946e79b0f79373fb7228cea85868f18a101d6360fe027742b3f4d7481 |
| inputs.baseline_covered_inputs.cache/features/text_features.h5.path | cache/features/text_features.h5 |
| inputs.baseline_covered_inputs.cache/features/text_features.h5.size_bytes | 154883968 |
| inputs.baseline_covered_inputs.cache/features/text_features.h5.sha256 | 82c0d10195121bcefbf11e2adfab03348377e5aae6a7288e76568b8da25617a1 |
| inputs.baseline_covered_inputs.cache/features/text_index.csv.path | cache/features/text_index.csv |
| inputs.baseline_covered_inputs.cache/features/text_index.csv.size_bytes | 7258137 |
| inputs.baseline_covered_inputs.cache/features/text_index.csv.sha256 | bc4248b529d9d9c6241b16bc871fe4ce1d43fab6c22213501d852989acaf43a1 |
| inputs.baseline_covered_inputs.cache/manifests/random_testA.jsonl.path | cache/manifests/random_testA.jsonl |
| inputs.baseline_covered_inputs.cache/manifests/random_testA.jsonl.size_bytes | 1252029 |
| inputs.baseline_covered_inputs.cache/manifests/random_testA.jsonl.sha256 | 35bfd08298df560c7042ec8ebac3c6e6daf115b410d4cffc100d24dee937a1dd |
| inputs.baseline_covered_inputs.cache/manifests/random_testB.jsonl.path | cache/manifests/random_testB.jsonl |
| inputs.baseline_covered_inputs.cache/manifests/random_testB.jsonl.size_bytes | 1137120 |
| inputs.baseline_covered_inputs.cache/manifests/random_testB.jsonl.sha256 | 32de6c535df1b9679a931e131baef74483cdd06cdf959c23cc0d13a394782259 |
| inputs.baseline_covered_inputs.cache/manifests/same_category_testA.jsonl.path | cache/manifests/same_category_testA.jsonl |
| inputs.baseline_covered_inputs.cache/manifests/same_category_testA.jsonl.size_bytes | 1313457 |
| inputs.baseline_covered_inputs.cache/manifests/same_category_testA.jsonl.sha256 | c5425b202ddb9a82db0715255a10e2c6b59e6dd66662dfd1c3c8caebcf67a8ee |
| inputs.baseline_covered_inputs.cache/manifests/same_category_testB.jsonl.path | cache/manifests/same_category_testB.jsonl |
| inputs.baseline_covered_inputs.cache/manifests/same_category_testB.jsonl.size_bytes | 1196525 |
| inputs.baseline_covered_inputs.cache/manifests/same_category_testB.jsonl.sha256 | 876d671dab5174532ff5799506bfffe32fdee95ac63d1bfdafcfd6616e2ead5e |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/random_testA.jsonl.path | cache/manifests_detr/manifests/random_testA.jsonl |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/random_testA.jsonl.size_bytes | 1267143 |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/random_testA.jsonl.sha256 | 176837d2ba53712e95e8edf9ba71a98a456d0d9814d5b682fd5b924f35fd388c |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/random_testB.jsonl.path | cache/manifests_detr/manifests/random_testB.jsonl |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/random_testB.jsonl.size_bytes | 1127679 |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/random_testB.jsonl.sha256 | 641b30aba5405ec657e6a99f92034706f4f3b00ef36c453ca0e132eec281ab70 |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/same_category_testA.jsonl.path | cache/manifests_detr/manifests/same_category_testA.jsonl |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/same_category_testA.jsonl.size_bytes | 1343514 |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/same_category_testA.jsonl.sha256 | 604a73f38248561c41b30864286395347ed41fbbc0e31f1f0a306c591f85727b |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/same_category_testB.jsonl.path | cache/manifests_detr/manifests/same_category_testB.jsonl |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/same_category_testB.jsonl.size_bytes | 1187573 |
| inputs.baseline_covered_inputs.cache/manifests_detr/manifests/same_category_testB.jsonl.sha256 | fcb3a2ff43b6b8a23e8fccd2e05aad4605bd507cfbc692a7a3fb94d26272bc4c |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_1/model.npz.path | results/phase0b_independent/seed_1/model.npz |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_1/model.npz.size_bytes | 858590 |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_1/model.npz.sha256 | 396d90da2f4f402e0dba86f4956c12efdf19b6b0c76a542e5b60a281810a8e74 |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_1/training.json.path | results/phase0b_independent/seed_1/training.json |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_1/training.json.size_bytes | 2452 |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_1/training.json.sha256 | b680a6784092d10c9a7408bffcba57a1b7cf498b41bac8b341fc808914598ea1 |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_2/model.npz.path | results/phase0b_independent/seed_2/model.npz |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_2/model.npz.size_bytes | 858590 |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_2/model.npz.sha256 | 6092592e6eef3cc031f9f3c0a33fcd296ea31061be80ab5008ceb8c6d7c910c4 |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_2/training.json.path | results/phase0b_independent/seed_2/training.json |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_2/training.json.size_bytes | 2558 |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_2/training.json.sha256 | 9dc0bab998b71aab3cc2c969217acf7d6ab218dc401a4edfd7db493b943408d2 |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_3/model.npz.path | results/phase0b_independent/seed_3/model.npz |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_3/model.npz.size_bytes | 858590 |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_3/model.npz.sha256 | b7671881b4c441d0ab01da1ee91572d2c3d66067c2a131e29e700460217d14ab |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_3/training.json.path | results/phase0b_independent/seed_3/training.json |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_3/training.json.size_bytes | 2347 |
| inputs.baseline_covered_inputs.results/phase0b_independent/seed_3/training.json.sha256 | 7ea10c653636d4645ec9ad6ab9126940bc38cf1ae0a64ee6d9d1453240a7e43f |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard5__b3_seed1.npz.path | results/phase1f_hard_semantic/predictions/hard5__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard5__b3_seed1.npz.size_bytes | 488639 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard5__b3_seed1.npz.sha256 | eeb01bccd298c5bc1399b5ad47c2d20860c0832a1bd3c86c197bd6411b02bb67 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand5__b3_seed1.npz.path | results/phase1f_hard_semantic/predictions/rand5__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand5__b3_seed1.npz.size_bytes | 492564 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand5__b3_seed1.npz.sha256 | 535ce06d00727ddf75d85ac9d8168023495733c9df67f9d01c24ae03c28cd189 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard5__b3_seed2.npz.path | results/phase1f_hard_semantic/predictions/hard5__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard5__b3_seed2.npz.size_bytes | 490808 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard5__b3_seed2.npz.sha256 | c04ed006bd622baff749929abe971eb6e48801c118f20df72065c89779fa81c8 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand5__b3_seed2.npz.path | results/phase1f_hard_semantic/predictions/rand5__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand5__b3_seed2.npz.size_bytes | 495849 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand5__b3_seed2.npz.sha256 | 553ae678e3f25a39ab3b5dfebe5ccca8af1f977e6338128fa567d63af86f93bc |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard5__b3_seed3.npz.path | results/phase1f_hard_semantic/predictions/hard5__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard5__b3_seed3.npz.size_bytes | 491673 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard5__b3_seed3.npz.sha256 | df1b2366f6540ff2b7dd8cb0cf49a44c47ee07b6309a9d65b299a2cd4e1c1df0 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand5__b3_seed3.npz.path | results/phase1f_hard_semantic/predictions/rand5__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand5__b3_seed3.npz.size_bytes | 496626 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand5__b3_seed3.npz.sha256 | 24965e381f0d58b34ec329f33deff2b626a1dbff7c40324a7505ff62ce0117e1 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard10__b3_seed1.npz.path | results/phase1f_hard_semantic/predictions/hard10__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard10__b3_seed1.npz.size_bytes | 461545 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard10__b3_seed1.npz.sha256 | 19d56d015396caadb15a7c62737b0af3bd93ec5bd428948bc2204db8d50f4f91 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand10__b3_seed1.npz.path | results/phase1f_hard_semantic/predictions/rand10__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand10__b3_seed1.npz.size_bytes | 466178 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand10__b3_seed1.npz.sha256 | 3e862e196843aae58a7ba90328e863fe55ce86d84719dba33392d962233eaeb6 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard10__b3_seed2.npz.path | results/phase1f_hard_semantic/predictions/hard10__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard10__b3_seed2.npz.size_bytes | 464545 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard10__b3_seed2.npz.sha256 | 05922bc114132d58dbb5c71bea10b8a74a6df92948a1a3ce8d935a79ab287c9c |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand10__b3_seed2.npz.path | results/phase1f_hard_semantic/predictions/rand10__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand10__b3_seed2.npz.size_bytes | 470154 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand10__b3_seed2.npz.sha256 | 5a69d81380016012ceb44cb3ee2b2a98340156ae7205153014afd304faa90454 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard10__b3_seed3.npz.path | results/phase1f_hard_semantic/predictions/hard10__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard10__b3_seed3.npz.size_bytes | 465756 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/hard10__b3_seed3.npz.sha256 | 767eeab700119876be39ee8a77a1298af6499e6a819c84aec2b2c825f6f3682d |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand10__b3_seed3.npz.path | results/phase1f_hard_semantic/predictions/rand10__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand10__b3_seed3.npz.size_bytes | 471045 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/rand10__b3_seed3.npz.sha256 | a8736cd10b1734c5a2087a439371a0e54a41b31578c5e895621e88b16c778caa |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m0__b3_seed1.npz.path | results/phase1f_hard_semantic/predictions/expb_m0__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m0__b3_seed1.npz.size_bytes | 512453 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m0__b3_seed1.npz.sha256 | 585d51d906282c6b4f816357953f59908c592cb6c633d9f296055093f5b05ef0 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m0__b3_seed2.npz.path | results/phase1f_hard_semantic/predictions/expb_m0__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m0__b3_seed2.npz.size_bytes | 516573 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m0__b3_seed2.npz.sha256 | d590e650744e1e24c6743ae2f85b57d5abdde313d1d07a3c4d05b1dbee080fc1 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m0__b3_seed3.npz.path | results/phase1f_hard_semantic/predictions/expb_m0__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m0__b3_seed3.npz.size_bytes | 517703 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m0__b3_seed3.npz.sha256 | bd1c11ce9ec056f0b99a54ee98c0d05a68452688df5c22612efc5c0e388f42bb |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m2__b3_seed1.npz.path | results/phase1f_hard_semantic/predictions/expb_m2__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m2__b3_seed1.npz.size_bytes | 510602 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m2__b3_seed1.npz.sha256 | f6159962c4a22396dcfcdeec13b4f21b7a6b9e44cf9227fd2437e17aaaf23169 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m2__b3_seed2.npz.path | results/phase1f_hard_semantic/predictions/expb_m2__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m2__b3_seed2.npz.size_bytes | 514551 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m2__b3_seed2.npz.sha256 | 3311faeb59a840a0b9b18c4e199431fdb677910902185bd1e088f4b6de843f13 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m2__b3_seed3.npz.path | results/phase1f_hard_semantic/predictions/expb_m2__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m2__b3_seed3.npz.size_bytes | 515814 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m2__b3_seed3.npz.sha256 | 97bb7545b87012633393318984442a9ea09ebe023739838a278c08180b9fc93c |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m4__b3_seed1.npz.path | results/phase1f_hard_semantic/predictions/expb_m4__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m4__b3_seed1.npz.size_bytes | 508989 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m4__b3_seed1.npz.sha256 | c389d470983ed8abe9fdfd68a74fb09e4bbb425a67516cd404ff44c2f7b8866c |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m4__b3_seed2.npz.path | results/phase1f_hard_semantic/predictions/expb_m4__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m4__b3_seed2.npz.size_bytes | 512831 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m4__b3_seed2.npz.sha256 | 4500da5acb71103c0f3e87e013fee6f4e30fbce1ffa934a5e725bd54d95aeb4b |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m4__b3_seed3.npz.path | results/phase1f_hard_semantic/predictions/expb_m4__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m4__b3_seed3.npz.size_bytes | 514200 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m4__b3_seed3.npz.sha256 | 95946dd44fdb53d768c99a7f8ba4bed00c48bc1dcfeb6114a5d465a10e40d2e1 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m8__b3_seed1.npz.path | results/phase1f_hard_semantic/predictions/expb_m8__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m8__b3_seed1.npz.size_bytes | 506095 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m8__b3_seed1.npz.sha256 | ce43d52108ed5c882bb0ede61fc6d519f9aded737c92103f3ab5150ed6a7ca21 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m8__b3_seed2.npz.path | results/phase1f_hard_semantic/predictions/expb_m8__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m8__b3_seed2.npz.size_bytes | 509544 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m8__b3_seed2.npz.sha256 | b534fec8797ecd03ee9e004b6116eef9466ad0a18e326c88932f16b0cf97bdd2 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m8__b3_seed3.npz.path | results/phase1f_hard_semantic/predictions/expb_m8__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m8__b3_seed3.npz.size_bytes | 510923 |
| inputs.baseline_covered_inputs.results/phase1f_hard_semantic/predictions/expb_m8__b3_seed3.npz.sha256 | b325c0236226b917ea0980c998d5c8ff83e8490f823943ce9aaeb8f0eb8bcefb |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed1.npz.path | results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed1.npz.size_bytes | 1773140 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed1.npz.sha256 | 048b2c5bbc620488496a513526e770bf2820776dfd39f2ef88d01969b36f657d |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed1.npz.path | results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed1.npz.size_bytes | 1920708 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed1.npz.sha256 | 0f721d6656bdb24fa464194dc144cf24394b32aab7515cd7356a9508070bcbbe |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed2.npz.path | results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed2.npz.size_bytes | 1773140 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed2.npz.sha256 | 1acaf8454d8b41edde7784d272234e7d77bbdcb48423ea70b8e8a9c947917ae5 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed2.npz.path | results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed2.npz.size_bytes | 1920708 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed2.npz.sha256 | bc3b78e7e6df70a878fd1b7da2a5ba0ec6f6f974c84e4c6c1e77884d7e59e847 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed3.npz.path | results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed3.npz.size_bytes | 1773140 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed3.npz.sha256 | 0555bd968db47cb0a048305bb5a1ccf384e9d56943422fd816386a9e7965e516 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed3.npz.path | results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed3.npz.size_bytes | 1920708 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__RPN__random_k5__b3_seed3.npz.sha256 | b70f7b6ce9e9de227925422dcaeb42b7c1f157eada1c53246940d118f38cd751 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed1.npz.path | results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed1.npz.size_bytes | 1543692 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed1.npz.sha256 | 392453e8d549d917f765c169e9979c923178a1eac309cb26a6b5736af8b69cab |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed1.npz.path | results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed1.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed1.npz.size_bytes | 1953092 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed1.npz.sha256 | 335c34a29a9450f2f75d9c427fce1f5105ffc1c937445f55a7892f0926b5a523 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed2.npz.path | results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed2.npz.size_bytes | 1543692 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed2.npz.sha256 | a3ff9b73fcd566ec80c582d9fb3fa6a914c4a612d4529a5abbe5b4dc2e1d24de |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed2.npz.path | results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed2.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed2.npz.size_bytes | 1953092 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed2.npz.sha256 | f0e1391b57f1943ce606ff0ae128e3c897d34deba883037bc995f9257a907f12 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed3.npz.path | results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed3.npz.size_bytes | 1543692 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed3.npz.sha256 | 9d20f4fba2ddf600c71fd1d492e279703291c52b9c1675a19fdcddbae275c300 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed3.npz.path | results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed3.npz |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed3.npz.size_bytes | 1953092 |
| inputs.baseline_covered_inputs.results/v2_proposal_robustness/predictions/p1__DETR__random_k5__b3_seed3.npz.sha256 | 8464221ea8360e1c37b3a0a4bc144ff9142a8d14662266f7a039eaad3e0a5fd8 |
| inputs.candidate_coco_supplemental_manifest | results/research_repair_v1/candidates/supplemental_input_manifest.json |
| inputs.candidate_cooco_sha256 | 777ef504abfeba349930485bb32ecd5b6529129a3e010f7852e1437cb049b732 |
| inputs.detr_features_supplemental_manifest | results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/forward_preflight_rows2_20261003T155906_55728/supplemental_input_manifest.json |
| inputs.detr_features_manifest_sha256 | 7b57d233258f742aae296f67514275c5a4750b9e7923240c9dbb2f9421bd54df |
| device | cuda |
| device_name | NVIDIA GeForce RTX 4060 Laptop GPU |
| forward_seconds | 64.295 |
| new_training | 0 |
| new_feature_extraction | 0 |
| model_refits | 0 |
| generated_utc | 2026-10-03T08:18:41Z |
| peak_vram_bytes | 12285440 |

## audit_run_20261003_Cslot_audit4/mismatch_by_category.csv

Source: `results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/mismatch_by_category.csv`; SHA256 `3547f2cb49b6213e80e47aa72b186c1aa409307d0103b96903b17a0768cd42ed`.

| family | true_target_category_id | true_target_category_name | n_expressions | n_mismatches | mismatch_rate | n_unknown_target_proposal_category |
| --- | --- | --- | --- | --- | --- | --- |
| DETR | 1 | person | 5334 | 0 | 0.0 | 0 |
| DETR | 2 | bicycle | 14 | 0 | 0.0 | 0 |
| DETR | 3 | car | 128 | 0 | 0.0 | 0 |
| DETR | 4 | motorcycle | 161 | 0 | 0.0 | 0 |
| DETR | 5 | airplane | 49 | 0 | 0.0 | 0 |
| DETR | 6 | bus | 123 | 0 | 0.0 | 0 |
| DETR | 7 | train | 78 | 0 | 0.0 | 0 |
| DETR | 8 | truck | 94 | 6 | 0.06382978723404255 | 0 |
| DETR | 9 | boat | 56 | 0 | 0.0 | 0 |
| DETR | 10 | traffic light | 18 | 0 | 0.0 | 0 |
| DETR | 11 | fire hydrant | 6 | 0 | 0.0 | 0 |
| DETR | 14 | parking meter | 55 | 0 | 0.0 | 0 |
| DETR | 15 | bench | 22 | 0 | 0.0 | 0 |
| DETR | 16 | bird | 82 | 0 | 0.0 | 0 |
| DETR | 17 | cat | 119 | 0 | 0.0 | 0 |
| DETR | 18 | dog | 76 | 0 | 0.0 | 0 |
| DETR | 19 | horse | 131 | 0 | 0.0 | 0 |
| DETR | 20 | sheep | 156 | 0 | 0.0 | 0 |
| DETR | 21 | cow | 132 | 6 | 0.045454545454545456 | 0 |
| DETR | 22 | elephant | 174 | 0 | 0.0 | 0 |
| DETR | 23 | bear | 64 | 0 | 0.0 | 0 |
| DETR | 24 | zebra | 141 | 0 | 0.0 | 0 |
| DETR | 25 | giraffe | 225 | 0 | 0.0 | 0 |
| DETR | 27 | backpack | 10 | 2 | 0.2 | 0 |
| DETR | 28 | umbrella | 70 | 0 | 0.0 | 0 |
| DETR | 31 | handbag | 9 | 0 | 0.0 | 0 |
| DETR | 32 | tie | 20 | 0 | 0.0 | 0 |
| DETR | 33 | suitcase | 152 | 0 | 0.0 | 0 |
| DETR | 36 | snowboard | 16 | 0 | 0.0 | 0 |
| DETR | 38 | kite | 11 | 0 | 0.0 | 0 |
| DETR | 40 | baseball glove | 3 | 0 | 0.0 | 0 |
| DETR | 42 | surfboard | 12 | 0 | 0.0 | 0 |
| DETR | 43 | tennis racket | 11 | 0 | 0.0 | 0 |
| DETR | 44 | bottle | 101 | 0 | 0.0 | 0 |
| DETR | 46 | wine glass | 34 | 0 | 0.0 | 0 |
| DETR | 47 | cup | 96 | 3 | 0.03125 | 0 |
| DETR | 51 | bowl | 246 | 6 | 0.024390243902439025 | 0 |
| DETR | 52 | banana | 137 | 0 | 0.0 | 0 |
| DETR | 53 | apple | 49 | 0 | 0.0 | 0 |
| DETR | 54 | sandwich | 192 | 3 | 0.015625 | 0 |
| DETR | 55 | orange | 92 | 0 | 0.0 | 0 |
| DETR | 56 | broccoli | 82 | 0 | 0.0 | 0 |
| DETR | 57 | carrot | 49 | 0 | 0.0 | 0 |
| DETR | 58 | hot dog | 56 | 0 | 0.0 | 0 |
| DETR | 59 | pizza | 169 | 0 | 0.0 | 0 |
| DETR | 60 | donut | 170 | 0 | 0.0 | 0 |
| DETR | 61 | cake | 99 | 0 | 0.0 | 0 |
| DETR | 62 | chair | 197 | 8 | 0.04060913705583756 | 0 |
| DETR | 63 | couch | 131 | 15 | 0.11450381679389313 | 0 |
| DETR | 64 | potted plant | 69 | 0 | 0.0 | 0 |
| DETR | 65 | bed | 111 | 0 | 0.0 | 0 |
| DETR | 67 | dining table | 63 | 0 | 0.0 | 0 |
| DETR | 70 | toilet | 70 | 0 | 0.0 | 0 |
| DETR | 72 | tv | 105 | 0 | 0.0 | 0 |
| DETR | 73 | laptop | 51 | 0 | 0.0 | 0 |
| DETR | 75 | remote | 14 | 0 | 0.0 | 0 |
| DETR | 76 | keyboard | 14 | 0 | 0.0 | 0 |
| DETR | 77 | cell phone | 25 | 0 | 0.0 | 0 |
| DETR | 78 | microwave | 15 | 0 | 0.0 | 0 |
| DETR | 79 | oven | 32 | 0 | 0.0 | 0 |
| DETR | 82 | refrigerator | 38 | 0 | 0.0 | 0 |
| DETR | 84 | book | 55 | 0 | 0.0 | 0 |
| DETR | 85 | clock | 40 | 0 | 0.0 | 0 |
| DETR | 86 | vase | 69 | 0 | 0.0 | 0 |
| DETR | 88 | teddy bear | 166 | 3 | 0.018072289156626505 | 0 |
| DETR | 90 | toothbrush | 12 | 0 | 0.0 | 0 |
| RPN | 1 | person | 5308 | 6 | 0.0011303692539562924 | 0 |
| RPN | 2 | bicycle | 14 | 0 | 0.0 | 0 |
| RPN | 3 | car | 128 | 6 | 0.046875 | 0 |
| RPN | 4 | motorcycle | 161 | 0 | 0.0 | 0 |
| RPN | 5 | airplane | 46 | 0 | 0.0 | 0 |
| RPN | 6 | bus | 115 | 0 | 0.0 | 0 |
| RPN | 7 | train | 78 | 0 | 0.0 | 0 |
| RPN | 8 | truck | 92 | 6 | 0.06521739130434782 | 0 |
| RPN | 9 | boat | 56 | 0 | 0.0 | 0 |
| RPN | 10 | traffic light | 14 | 0 | 0.0 | 0 |
| RPN | 11 | fire hydrant | 6 | 0 | 0.0 | 0 |
| RPN | 14 | parking meter | 55 | 0 | 0.0 | 0 |
| RPN | 15 | bench | 22 | 0 | 0.0 | 0 |
| RPN | 16 | bird | 82 | 0 | 0.0 | 0 |
| RPN | 17 | cat | 119 | 0 | 0.0 | 0 |
| RPN | 18 | dog | 76 | 0 | 0.0 | 0 |
| RPN | 19 | horse | 131 | 0 | 0.0 | 0 |
| RPN | 20 | sheep | 156 | 0 | 0.0 | 0 |
| RPN | 21 | cow | 132 | 6 | 0.045454545454545456 | 0 |
| RPN | 22 | elephant | 174 | 0 | 0.0 | 0 |
| RPN | 23 | bear | 64 | 3 | 0.046875 | 0 |
| RPN | 24 | zebra | 141 | 0 | 0.0 | 0 |
| RPN | 25 | giraffe | 222 | 0 | 0.0 | 0 |
| RPN | 27 | backpack | 10 | 0 | 0.0 | 0 |
| RPN | 28 | umbrella | 57 | 0 | 0.0 | 0 |
| RPN | 31 | handbag | 9 | 0 | 0.0 | 0 |
| RPN | 32 | tie | 20 | 0 | 0.0 | 0 |
| RPN | 33 | suitcase | 152 | 0 | 0.0 | 0 |
| RPN | 36 | snowboard | 16 | 0 | 0.0 | 0 |
| RPN | 38 | kite | 11 | 0 | 0.0 | 0 |
| RPN | 40 | baseball glove | 3 | 0 | 0.0 | 0 |
| RPN | 42 | surfboard | 12 | 0 | 0.0 | 0 |
| RPN | 43 | tennis racket | 11 | 3 | 0.2727272727272727 | 0 |
| RPN | 44 | bottle | 101 | 0 | 0.0 | 0 |
| RPN | 46 | wine glass | 34 | 0 | 0.0 | 0 |
| RPN | 47 | cup | 96 | 3 | 0.03125 | 0 |
| RPN | 51 | bowl | 234 | 9 | 0.038461538461538464 | 0 |
| RPN | 52 | banana | 132 | 0 | 0.0 | 0 |
| RPN | 53 | apple | 46 | 0 | 0.0 | 0 |
| RPN | 54 | sandwich | 186 | 6 | 0.03225806451612903 | 0 |
| RPN | 55 | orange | 91 | 0 | 0.0 | 0 |
| RPN | 56 | broccoli | 85 | 0 | 0.0 | 0 |
| RPN | 57 | carrot | 48 | 0 | 0.0 | 0 |
| RPN | 58 | hot dog | 56 | 0 | 0.0 | 0 |
| RPN | 59 | pizza | 159 | 0 | 0.0 | 0 |
| RPN | 60 | donut | 166 | 0 | 0.0 | 0 |
| RPN | 61 | cake | 99 | 0 | 0.0 | 0 |
| RPN | 62 | chair | 184 | 11 | 0.059782608695652176 | 0 |
| RPN | 63 | couch | 124 | 15 | 0.12096774193548387 | 0 |
| RPN | 64 | potted plant | 63 | 0 | 0.0 | 0 |
| RPN | 65 | bed | 111 | 6 | 0.05405405405405406 | 0 |
| RPN | 67 | dining table | 38 | 0 | 0.0 | 0 |
| RPN | 70 | toilet | 70 | 0 | 0.0 | 0 |
| RPN | 72 | tv | 97 | 3 | 0.030927835051546393 | 0 |
| RPN | 73 | laptop | 51 | 0 | 0.0 | 0 |
| RPN | 75 | remote | 14 | 0 | 0.0 | 0 |
| RPN | 76 | keyboard | 14 | 0 | 0.0 | 0 |
| RPN | 77 | cell phone | 25 | 0 | 0.0 | 0 |
| RPN | 78 | microwave | 15 | 0 | 0.0 | 0 |
| RPN | 79 | oven | 28 | 0 | 0.0 | 0 |
| RPN | 82 | refrigerator | 38 | 0 | 0.0 | 0 |
| RPN | 84 | book | 46 | 0 | 0.0 | 0 |
| RPN | 85 | clock | 40 | 0 | 0.0 | 0 |
| RPN | 86 | vase | 69 | 0 | 0.0 | 0 |
| RPN | 88 | teddy bear | 160 | 3 | 0.01875 | 0 |
| RPN | 90 | toothbrush | 12 | 0 | 0.0 | 0 |

## audit_run_20261003_Cslot_audit4/source_cohort_mismatch_coverage.csv

Source: `results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/source_cohort_mismatch_coverage.csv`; SHA256 `37870a7374cb2f768b53d4c7651c0b040ec3cde6ecd80b73f9c97d6058c72ec9`.

| family | cell | candidate_source | source_path | source_rows | source_sentence_ids_unique | source_rows_covered_by_category_audit | target_proposal_mismatches | target_proposal_mismatch_rate | measured_cell_applicable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RPN | hard5 | phase1f_frozen_candidate_archive | results/phase1f_hard_semantic/manifests/hard5_candidates.npz | 9487 | True | 9487 | 54 | 0.005691999578370402 | True |
| RPN | hard10 | phase1f_frozen_candidate_archive | results/phase1f_hard_semantic/manifests/hard10_candidates.npz | 6765 | True | 6765 | 24 | 0.003547671840354767 | True |
| RPN | expb_m0 | phase1f_frozen_candidate_archive | results/phase1f_hard_semantic/manifests/expb_m0_candidates.npz | 7410 | True | 7410 | 27 | 0.0036437246963562753 | True |
| RPN | expb_m2 | phase1f_frozen_candidate_archive | results/phase1f_hard_semantic/manifests/expb_m2_candidates.npz | 7410 | True | 7410 | 27 | 0.0036437246963562753 | True |
| RPN | expb_m4 | phase1f_frozen_candidate_archive | results/phase1f_hard_semantic/manifests/expb_m4_candidates.npz | 7410 | True | 7410 | 27 | 0.0036437246963562753 | True |
| RPN | expb_m8 | phase1f_frozen_candidate_archive | results/phase1f_hard_semantic/manifests/expb_m8_candidates.npz | 7410 | True | 7410 | 27 | 0.0036437246963562753 | True |
| RPN | hard5 | v2_p1_frozen_hard_k5_predictions | results/v2_proposal_robustness/predictions/p1__RPN__hard_k5__b3_seed1.npz | 9623 | True | 9623 | 54 | 0.005611555647926842 | True |
| DETR | hard5 | v2_p1_frozen_hard_k5_predictions | results/v2_proposal_robustness/predictions/p1__DETR__hard_k5__b3_seed1.npz | 8376 | True | 8376 | 11 | 0.0013132760267430754 | True |

## audit_run_20261003_Cslot_audit4/summary.json

Source: `results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/summary.json`; SHA256 `d1a7eecac75467a7a46bae3133448451c57aa7a0c8550a4aca1da3cf3fe6cce8`.

| Field | Stored value |
|---|---|
| schema | research-repair-v1-candidate-summary-v1 |
| artifact | research_repair_v1_candidate_category_audit |
| audit_run_id | 20261003_Cslot_audit4 |
| protocol | Research Repair v1 C |
| status | AUDIT_COMPLETE_MISMATCH_FOUND |
| candidate_variant | true_target_category_v1 |
| candidate_variant_constructed | True |
| candidate_variant_reason | at least one target proposal category differs from its RefCOCO+ ann_id category |
| inputs.refs | data/raw/refcoco+/refcoco+/refs(unc).p |
| inputs.instances | data/raw/refcoco+/refcoco+/instances.json |
| inputs.coco_gt | data/raw/annotations/instances_train2014.json |
| inputs.coco_gt_sha256 | 777ef504abfeba349930485bb32ecd5b6529129a3e010f7852e1437cb049b732 |
| inputs.baseline_n_files | 794 |
| inputs.supplemental_manifest | results/research_repair_v1/candidates/supplemental_input_manifest.json |
| inputs.prepare_path_gap.prepared_expected_path | data/raw/mscoco/annotations/instances_train2014.json |
| inputs.prepare_path_gap.prepared_expected_path_exists | False |
| inputs.prepare_path_gap.actual_source_path | data/raw/annotations/instances_train2014.json |
| inputs.prepare_path_gap.reason | the candidate audit loads the repository's actual COCO GT path; it was not covered by the prepared baseline |
| inputs.iou_threshold | 0.5 |
| inputs.target_equivalence_iou_threshold | 0.5 |
| inputs.proposal_category_assignment | best-IoU COCO object over the full frozen GT object list, including crowd rows; category is UNKNOWN/-1 below IoU 0.5 |
| inputs.candidate_families | ["RPN", "DETR"] |
| inputs.candidate_cells.RPN | ["hard5", "hard10", "expb_m0", "expb_m2", "expb_m4", "expb_m8"] |
| inputs.candidate_cells.DETR | ["hard5"] |
| inputs.frozen_seed_identity_checked | [1, 2, 3] |
| inputs.new_feature_extraction | 0 |
| inputs.new_training | 0 |
| inputs.new_model_forward | 0 |
| inputs.GPU_used | False |
| mismatch.total_expressions | 21026 |
| mismatch.total_mismatches | 138 |
| mismatch.overall_rate | 0.006563302577760868 |
| mismatch.by_family.RPN.n_expressions | 10425 |
| mismatch.by_family.RPN.n_mismatches | 86 |
| mismatch.by_family.RPN.mismatch_rate | 0.008249400479616307 |
| mismatch.by_family.RPN.n_unknown_target_proposal_category | 0 |
| mismatch.by_family.DETR.n_expressions | 10601 |
| mismatch.by_family.DETR.n_mismatches | 52 |
| mismatch.by_family.DETR.mismatch_rate | 0.004905197622865768 |
| mismatch.by_family.DETR.n_unknown_target_proposal_category | 0 |
| mismatch.summary_dimensions | ["expression", "true_category", "image", "target_GT_object_ann_id", "proposal_assigned_category/object"] |
| mismatch.unmatched_wording | proposal not assigned to a COCO object at IoU >= 0.5; never described as nothing |
| historical_candidate_source_scope.family_cells.RPN | ["hard5", "hard10", "expb_m0", "expb_m2", "expb_m4", "expb_m8"] |
| historical_candidate_source_scope.family_cells.DETR | ["hard5"] |
| historical_candidate_source_scope.source_cohorts_kept_separate | True |
| historical_candidate_source_scope.mismatch_audit_universe | frozen P1 random K5 rows; all frozen hard/dose source cohort sentence_id/ref_id/image_id rows verified as subsets and summarized in source_cohort_mismatch_coverage.csv |
| candidate_geometry.unmatched_wording | proposal not assigned to a COCO object at IoU >= 0.5; never described as nothing |
| candidate_checks.RPN/dose_common_m8.old_available | 7528 |
| candidate_checks.RPN/dose_common_m8.new_available | 7534 |
| candidate_checks.RPN/dose_common_m8.intersection | 7507 |
| candidate_checks.RPN/all_dose_intersection.rows | 7507 |
| candidate_checks.RPN/all_dose_intersection.definition | P1 K10 identity rows old/new candidate-supply available at every m in {0,2,4,8}; measured source cohorts are separately enumerated |
| output_files.expression_audit | mismatch_by_expression.csv |
| output_files.category_summary | mismatch_by_category.csv |
| output_files.category_confusion | mismatch_category_confusion.csv |
| output_files.image_summary | mismatch_by_image.csv |
| output_files.object_summary | mismatch_by_object.csv |
| output_files.availability_intersection | availability_intersection.csv |
| output_files.candidate_source_availability | source_candidate_availability.csv |
| output_files.source_mismatch_coverage | source_cohort_mismatch_coverage.csv |
| output_files.candidate_geometry_summary | candidate_geometry_historical_source_summary.csv |
| output_files.candidate_geometry_per_expression | candidate_geometry_historical_source_per_expression.csv |
| output_files.candidate_geometry_supply_intersection_summary | candidate_geometry_supply_intersection_summary.csv |
| output_files.candidate_geometry_supply_intersection_per_expression | candidate_geometry_supply_intersection_per_expression.csv |
| output_files.candidate_geometry_sensitivity_summary | candidate_geometry_sensitivity_summary.csv |
| output_files.candidate_geometry_sensitivity_per_expression | candidate_geometry_sensitivity_per_expression.csv |
| output_files.versioned_candidate_indices | results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/candidate_sets_true_target_category_v1.npz |
| output_artifacts.versioned_candidate_indices.path | results/research_repair_v1/candidates/audit_run_20261003_Cslot_audit4/candidate_sets_true_target_category_v1.npz |
| output_artifacts.versioned_candidate_indices.size_bytes | 1491524 |
| output_artifacts.versioned_candidate_indices.sha256 | 4f93780ddff4fb26fc1c8e3894ffba3dd172d96068394e8fb37672ede9f26b2d |
| generated_utc | 2026-10-03T07:23:05Z |

## availability_intersection.csv

Source: `results/research_repair_v1/candidates/availability_intersection.csv`; SHA256 `5127ebd2b68c98b456dc49d995ef4319702dda17ec0c47cd7d995dc66e28f2dc`.

| family | cell | k | cell_kind | same_category_proposals_required | evaluation_universe_rows | historical_old_available_rows | corrected_true_category_available_rows | old_new_common_rows | old_only_rows_lost_under_corrected_category | corrected_only_rows_not_in_historical_cohort | old_common_retention | new_common_retention | new_variant | comparison_cohort |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RPN | hard5 | 5 | hard | 4 | 10425 | 9623 | 9626 | 9602 | 21 | 24 | 0.9978177283591395 | 0.9975067525451901 | true_target_category_v1 | old_new_common_availability_intersection |
| RPN | hard10 | 10 | hard | 9 | 10425 | 6874 | 6880 | 6856 | 18 | 24 | 0.9973814372999709 | 0.9965116279069768 | true_target_category_v1 | old_new_common_availability_intersection |
| RPN | expb_m0 | 10 | dose | 0 | 10425 | 7528 | 10425 | 7528 | 0 | 2897 | 1.0 | 0.7221103117505995 | true_target_category_v1 | old_new_common_availability_intersection |
| RPN | expb_m2 | 10 | dose | 2 | 10425 | 7528 | 10187 | 7528 | 0 | 2659 | 1.0 | 0.7389810542848729 | true_target_category_v1 | old_new_common_availability_intersection |
| RPN | expb_m4 | 10 | dose | 4 | 10425 | 7528 | 9626 | 7516 | 12 | 2110 | 0.9984059511158342 | 0.7808019945979638 | true_target_category_v1 | old_new_common_availability_intersection |
| RPN | expb_m8 | 10 | dose | 8 | 10425 | 7528 | 7534 | 7507 | 21 | 27 | 0.9972104144527099 | 0.9964162463498806 | true_target_category_v1 | old_new_common_availability_intersection |
| DETR | hard5 | 5 | hard | 4 | 10601 | 8376 | 8396 | 8373 | 3 | 23 | 0.9996418338108882 | 0.9972606002858504 | true_target_category_v1 | old_new_common_availability_intersection |

## io_probe/basic.csv

Source: `results/research_repair_v1/candidates/io_probe/basic.csv`; SHA256 `a603d0cf912561fbeea14c1fb4abc59997d652e8ac075fc27450c7ca70b6c868`.

| a | b\n1 | 2\n |
| --- | --- | --- |

## io_probe/replace_target.csv

Source: `results/research_repair_v1/candidates/io_probe/replace_target.csv`; SHA256 `f7c88487cb23fdbe9b6f49b243a6a6ffe3d14cbbf51ae4c23117054a2ae7d87e`.

| atomic\n |
| --- |

## mismatch_by_category.csv

Source: `results/research_repair_v1/candidates/mismatch_by_category.csv`; SHA256 `3547f2cb49b6213e80e47aa72b186c1aa409307d0103b96903b17a0768cd42ed`.

| family | true_target_category_id | true_target_category_name | n_expressions | n_mismatches | mismatch_rate | n_unknown_target_proposal_category |
| --- | --- | --- | --- | --- | --- | --- |
| DETR | 1 | person | 5334 | 0 | 0.0 | 0 |
| DETR | 2 | bicycle | 14 | 0 | 0.0 | 0 |
| DETR | 3 | car | 128 | 0 | 0.0 | 0 |
| DETR | 4 | motorcycle | 161 | 0 | 0.0 | 0 |
| DETR | 5 | airplane | 49 | 0 | 0.0 | 0 |
| DETR | 6 | bus | 123 | 0 | 0.0 | 0 |
| DETR | 7 | train | 78 | 0 | 0.0 | 0 |
| DETR | 8 | truck | 94 | 6 | 0.06382978723404255 | 0 |
| DETR | 9 | boat | 56 | 0 | 0.0 | 0 |
| DETR | 10 | traffic light | 18 | 0 | 0.0 | 0 |
| DETR | 11 | fire hydrant | 6 | 0 | 0.0 | 0 |
| DETR | 14 | parking meter | 55 | 0 | 0.0 | 0 |
| DETR | 15 | bench | 22 | 0 | 0.0 | 0 |
| DETR | 16 | bird | 82 | 0 | 0.0 | 0 |
| DETR | 17 | cat | 119 | 0 | 0.0 | 0 |
| DETR | 18 | dog | 76 | 0 | 0.0 | 0 |
| DETR | 19 | horse | 131 | 0 | 0.0 | 0 |
| DETR | 20 | sheep | 156 | 0 | 0.0 | 0 |
| DETR | 21 | cow | 132 | 6 | 0.045454545454545456 | 0 |
| DETR | 22 | elephant | 174 | 0 | 0.0 | 0 |
| DETR | 23 | bear | 64 | 0 | 0.0 | 0 |
| DETR | 24 | zebra | 141 | 0 | 0.0 | 0 |
| DETR | 25 | giraffe | 225 | 0 | 0.0 | 0 |
| DETR | 27 | backpack | 10 | 2 | 0.2 | 0 |
| DETR | 28 | umbrella | 70 | 0 | 0.0 | 0 |
| DETR | 31 | handbag | 9 | 0 | 0.0 | 0 |
| DETR | 32 | tie | 20 | 0 | 0.0 | 0 |
| DETR | 33 | suitcase | 152 | 0 | 0.0 | 0 |
| DETR | 36 | snowboard | 16 | 0 | 0.0 | 0 |
| DETR | 38 | kite | 11 | 0 | 0.0 | 0 |
| DETR | 40 | baseball glove | 3 | 0 | 0.0 | 0 |
| DETR | 42 | surfboard | 12 | 0 | 0.0 | 0 |
| DETR | 43 | tennis racket | 11 | 0 | 0.0 | 0 |
| DETR | 44 | bottle | 101 | 0 | 0.0 | 0 |
| DETR | 46 | wine glass | 34 | 0 | 0.0 | 0 |
| DETR | 47 | cup | 96 | 3 | 0.03125 | 0 |
| DETR | 51 | bowl | 246 | 6 | 0.024390243902439025 | 0 |
| DETR | 52 | banana | 137 | 0 | 0.0 | 0 |
| DETR | 53 | apple | 49 | 0 | 0.0 | 0 |
| DETR | 54 | sandwich | 192 | 3 | 0.015625 | 0 |
| DETR | 55 | orange | 92 | 0 | 0.0 | 0 |
| DETR | 56 | broccoli | 82 | 0 | 0.0 | 0 |
| DETR | 57 | carrot | 49 | 0 | 0.0 | 0 |
| DETR | 58 | hot dog | 56 | 0 | 0.0 | 0 |
| DETR | 59 | pizza | 169 | 0 | 0.0 | 0 |
| DETR | 60 | donut | 170 | 0 | 0.0 | 0 |
| DETR | 61 | cake | 99 | 0 | 0.0 | 0 |
| DETR | 62 | chair | 197 | 8 | 0.04060913705583756 | 0 |
| DETR | 63 | couch | 131 | 15 | 0.11450381679389313 | 0 |
| DETR | 64 | potted plant | 69 | 0 | 0.0 | 0 |
| DETR | 65 | bed | 111 | 0 | 0.0 | 0 |
| DETR | 67 | dining table | 63 | 0 | 0.0 | 0 |
| DETR | 70 | toilet | 70 | 0 | 0.0 | 0 |
| DETR | 72 | tv | 105 | 0 | 0.0 | 0 |
| DETR | 73 | laptop | 51 | 0 | 0.0 | 0 |
| DETR | 75 | remote | 14 | 0 | 0.0 | 0 |
| DETR | 76 | keyboard | 14 | 0 | 0.0 | 0 |
| DETR | 77 | cell phone | 25 | 0 | 0.0 | 0 |
| DETR | 78 | microwave | 15 | 0 | 0.0 | 0 |
| DETR | 79 | oven | 32 | 0 | 0.0 | 0 |
| DETR | 82 | refrigerator | 38 | 0 | 0.0 | 0 |
| DETR | 84 | book | 55 | 0 | 0.0 | 0 |
| DETR | 85 | clock | 40 | 0 | 0.0 | 0 |
| DETR | 86 | vase | 69 | 0 | 0.0 | 0 |
| DETR | 88 | teddy bear | 166 | 3 | 0.018072289156626505 | 0 |
| DETR | 90 | toothbrush | 12 | 0 | 0.0 | 0 |
| RPN | 1 | person | 5308 | 6 | 0.0011303692539562924 | 0 |
| RPN | 2 | bicycle | 14 | 0 | 0.0 | 0 |
| RPN | 3 | car | 128 | 6 | 0.046875 | 0 |
| RPN | 4 | motorcycle | 161 | 0 | 0.0 | 0 |
| RPN | 5 | airplane | 46 | 0 | 0.0 | 0 |
| RPN | 6 | bus | 115 | 0 | 0.0 | 0 |
| RPN | 7 | train | 78 | 0 | 0.0 | 0 |
| RPN | 8 | truck | 92 | 6 | 0.06521739130434782 | 0 |
| RPN | 9 | boat | 56 | 0 | 0.0 | 0 |
| RPN | 10 | traffic light | 14 | 0 | 0.0 | 0 |
| RPN | 11 | fire hydrant | 6 | 0 | 0.0 | 0 |
| RPN | 14 | parking meter | 55 | 0 | 0.0 | 0 |
| RPN | 15 | bench | 22 | 0 | 0.0 | 0 |
| RPN | 16 | bird | 82 | 0 | 0.0 | 0 |
| RPN | 17 | cat | 119 | 0 | 0.0 | 0 |
| RPN | 18 | dog | 76 | 0 | 0.0 | 0 |
| RPN | 19 | horse | 131 | 0 | 0.0 | 0 |
| RPN | 20 | sheep | 156 | 0 | 0.0 | 0 |
| RPN | 21 | cow | 132 | 6 | 0.045454545454545456 | 0 |
| RPN | 22 | elephant | 174 | 0 | 0.0 | 0 |
| RPN | 23 | bear | 64 | 3 | 0.046875 | 0 |
| RPN | 24 | zebra | 141 | 0 | 0.0 | 0 |
| RPN | 25 | giraffe | 222 | 0 | 0.0 | 0 |
| RPN | 27 | backpack | 10 | 0 | 0.0 | 0 |
| RPN | 28 | umbrella | 57 | 0 | 0.0 | 0 |
| RPN | 31 | handbag | 9 | 0 | 0.0 | 0 |
| RPN | 32 | tie | 20 | 0 | 0.0 | 0 |
| RPN | 33 | suitcase | 152 | 0 | 0.0 | 0 |
| RPN | 36 | snowboard | 16 | 0 | 0.0 | 0 |
| RPN | 38 | kite | 11 | 0 | 0.0 | 0 |
| RPN | 40 | baseball glove | 3 | 0 | 0.0 | 0 |
| RPN | 42 | surfboard | 12 | 0 | 0.0 | 0 |
| RPN | 43 | tennis racket | 11 | 3 | 0.2727272727272727 | 0 |
| RPN | 44 | bottle | 101 | 0 | 0.0 | 0 |
| RPN | 46 | wine glass | 34 | 0 | 0.0 | 0 |
| RPN | 47 | cup | 96 | 3 | 0.03125 | 0 |
| RPN | 51 | bowl | 234 | 9 | 0.038461538461538464 | 0 |
| RPN | 52 | banana | 132 | 0 | 0.0 | 0 |
| RPN | 53 | apple | 46 | 0 | 0.0 | 0 |
| RPN | 54 | sandwich | 186 | 6 | 0.03225806451612903 | 0 |
| RPN | 55 | orange | 91 | 0 | 0.0 | 0 |
| RPN | 56 | broccoli | 85 | 0 | 0.0 | 0 |
| RPN | 57 | carrot | 48 | 0 | 0.0 | 0 |
| RPN | 58 | hot dog | 56 | 0 | 0.0 | 0 |
| RPN | 59 | pizza | 159 | 0 | 0.0 | 0 |
| RPN | 60 | donut | 166 | 0 | 0.0 | 0 |
| RPN | 61 | cake | 99 | 0 | 0.0 | 0 |
| RPN | 62 | chair | 184 | 11 | 0.059782608695652176 | 0 |
| RPN | 63 | couch | 124 | 15 | 0.12096774193548387 | 0 |
| RPN | 64 | potted plant | 63 | 0 | 0.0 | 0 |
| RPN | 65 | bed | 111 | 6 | 0.05405405405405406 | 0 |
| RPN | 67 | dining table | 38 | 0 | 0.0 | 0 |
| RPN | 70 | toilet | 70 | 0 | 0.0 | 0 |
| RPN | 72 | tv | 97 | 3 | 0.030927835051546393 | 0 |
| RPN | 73 | laptop | 51 | 0 | 0.0 | 0 |
| RPN | 75 | remote | 14 | 0 | 0.0 | 0 |
| RPN | 76 | keyboard | 14 | 0 | 0.0 | 0 |
| RPN | 77 | cell phone | 25 | 0 | 0.0 | 0 |
| RPN | 78 | microwave | 15 | 0 | 0.0 | 0 |
| RPN | 79 | oven | 28 | 0 | 0.0 | 0 |
| RPN | 82 | refrigerator | 38 | 0 | 0.0 | 0 |
| RPN | 84 | book | 46 | 0 | 0.0 | 0 |
| RPN | 85 | clock | 40 | 0 | 0.0 | 0 |
| RPN | 86 | vase | 69 | 0 | 0.0 | 0 |
| RPN | 88 | teddy bear | 160 | 3 | 0.01875 | 0 |
| RPN | 90 | toothbrush | 12 | 0 | 0.0 | 0 |

