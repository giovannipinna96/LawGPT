# main test for makedataset

from icecream import ic
from make_dataset import (
    MakeDataset_Admin_it,
    MakeDataset_Admin_it2,
    MakeDataset_EurLexSumIt,
    MakeDataset_EuroParl,
    MakeDataset_Itacasehold,
    MakeDataset_Costituzionale,
    MakeDataset_OneStopEnglishCorpus,
    MakeDataset_Paccss_it,
    MakeDataset_Simpa,
    MakeDataset_Simpitiki,
)

# # EURLEXSUMIT
# print("Starting create the dataset for EurLexSumIt...")
# eur_lex_sum_it = MakeDataset_EurLexSumIt("lawgpt/data/raw/summarization/eur-lex-sum_it")
# ic("print the name of the dataset")
# ic(eur_lex_sum_it.name)
# ic("create the dataset")
# eur_lex_sum_it.create_data("lawgpt/data/interim/summarization/eur-lex-sum_it")
# ic(f"correctly create the {eur_lex_sum_it.name} dataset")
# print("Finished create the dataset for EurLexSumIt...")

# # EUROPARL
# print("Staring create the dataset for Europarl...")
# euro_parl = MakeDataset_EuroParl(
#     "lawgpt/data/raw/summarization/europarl_corpus_it/europarl_it_eng"
# )
# ic(f"dataset name -> {euro_parl.name}")
# ic("create the dataset")
# euro_parl.create_data("lawgpt/data/interim/summarization/europarl")
# ic(f"correctly create the {euro_parl.name} dataset")
# print("Finished create the dataset for Europarl...")

# # ITACASEHOLD
# print("Starting create the dataset for Italcasehold...")
# itacasehold = MakeDataset_Itacasehold("lawgpt/data/raw/summarization/itacasehold")
# ic(f"dataset name -> {itacasehold.name}")
# ic("create the dataset")
# itacasehold.create_data("lawgpt/data/interim/summarization/itacasehold")
# ic(f"correctly create the {itacasehold.name} dataset")
# print("Finished create the dataset for Itacasehold...")


# COSTITUZIONALE
print("Starting create the dataset for Costituzionale...")
costitutional = MakeDataset_Costituzionale(
    "lawgpt/data/external/dati_corte_costituzionale"
)
# ic(f"dataset name -> {costitutional.name}")
# ic("create the dataset")
# costitutional.create_data("lawgpt/data/interim/summarization/costituzionale")
# ic(f"correctly create the {costitutional.name} dataset")
# print("Finished create the dataset for Costituzionale...")

costitutional.union_massime_and_pronunce(
    "lawgpt/data/interim/summarization/costituzionale/massime.json",
    "lawgpt/data/interim/summarization/costituzionale/pronuncie.json",
    "lawgpt/data/interim/summarization/costituzionale/massime_and_pronincie.json",
)

# # ADMIN_IT
# print("Starting create the dataset for Admin_it...")
# admin_it = MakeDataset_Admin_it("lawgpt/data/raw/simplification/admin-It")
# ic(f"dataset name -> {admin_it.name}")
# ic("create the dataset")
# admin_it.create_data("lawgpt/data/interim/simplification/admin_it")
# ic(f"correctly create the {admin_it.name} dataset")
# print("Finished create the dataset for Admin_it...")

# # ADMIN_IT2
# print("Starting create the dataset for Admin_it2...")
# admin_it2 = MakeDataset_Admin_it2("lawgpt/data/raw/simplification/admin-it-l2")
# ic(f"dataset name -> {admin_it2.name}")
# ic("create the dataset")
# admin_it2.create_data("lawgpt/data/interim/simplification/admin_it2")
# ic(f"correctly create the {admin_it2.name} dataset")
# print("Finished create the dataset for Admin_it2...")

# # OneStopEnglishCorpus
# print("Starting create the dataset for OneStopEnglishCorpus...")
# one_stop_english_corpus = MakeDataset_OneStopEnglishCorpus(
#     "lawgpt/data/raw/simplification/OneStopEnglishCorpus/Texts-SeparatedByReadingLevel"
# )
# ic(f"dataset name -> {one_stop_english_corpus.name}")
# ic("create the dataset")
# one_stop_english_corpus.create_data(
#     "lawgpt/data/interim/simplification/onestopenglishcorpus"
# )
# ic(f"correctly create the {one_stop_english_corpus.name} dataset")
# print("Finished create the dataset for OneStopEnglishCorpus...")

# # PACCSS_IT
# print("Starting create the dataset for Paccss_it...")
# paccss_it = MakeDataset_Paccss_it("lawgpt/data/raw/simplification/PaCCSS-IT/data-set")
# ic(f"dataset name -> {paccss_it.name}")
# ic("create the dataset")
# paccss_it.create_data("lawgpt/data/interim/simplification/paccss_it")
# ic(f"correctly create the {paccss_it.name} dataset")
# print("Finished create the dataset for Paccss_it...")

# # SIMPA
# print("Starting create the datset for Simpa...")
# simpa = MakeDataset_Simpa("lawgpt/data/raw/simplification/simpa")
# ic(f"dataset name -> {simpa.name}")
# ic("create the dataset")
# simpa.create_data("lawgpt/data/interim/simplification/simpa")
# ic(f"correctly create the {simpa.name} dataset")
# print("Finished create the dataset for Simpa...")

# # Simpitiki
# print("Starting create the datset for Simpitiki...")
# simpitiki = MakeDataset_Simpitiki("lawgpt/data/raw/simplification/simpitiki/corpus")
# ic(f"dataset name -> {simpitiki.name}")
# ic("create the dataset")
# simpitiki.create_data("lawgpt/data/interim/simplification/simpitiki")
# ic(f"correctly create the {simpitiki.name} dataset")
# print("Finished create the dataset for Simpitiki...")
