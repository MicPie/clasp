We use the ood and id valid dataset split outlined in the ProGen publication (https://www.biorxiv.org/content/10.1101/2020.03.07.982272v2).

Download pfam data:
```
wget http://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.regions.uniprot.tsv.gz
```

Get ProGen ood entries:
```
grep -f ../progen_ood_families.csv Pfam-A.regions.uniprot.tsv > Pfam-A_regions_uniprot_progen_ood_families.tsv
```

Get ProGen ood uniprot acc column:
```
cut -f1 Pfam-A_regions_uniprot_progen_ood_families.tsv > progen_ood_families_uniprot_acc.tsv
```

Get ProGen ood entries from csv data file into separate ood valid data file:
```
grep -f progen_ood_families_uniprot_acc.csv uniprot_full.csv > uniprot_full_valid-ood.csv
```

Remove ProGen ood entries from csv date file (https://stackoverflow.com/questions/28647088/grep-for-a-line-in-a-file-then-remove-the-line):
```
grep -v -f progen_ood_families_uniprot_acc.csv uniprot_full.csv > uniprot_full_wo-valid-ood.csv.tmp && mv uniprot_full_wo-valid-ood.csv.tmp uniprot_full_wo-valid-ood.csv
```

Test if no ood entries are still included:
```
grep -f progen_ood_families_uniprot_acc.csv uniprot_full_wo-valid-ood.csv | wc -l
```
should return 0

Get 0.5% subsample of the csv data file for id valid data file (https://stackoverflow.com/questions/19770404/random-split-files-with-specific-proportion, if needed install `gawk` with `sudo apt-get install gawk`):
```
gawk 'BEGIN {srand()} {f = FILENAME (rand() <= 0.995 ? ".995" : ".005"); print > f}' uniprot_full_wo-valid-ood.csv
mv uniprot_full_wo-valid-ood.csv.995 uniprot_full_wo-valid-ood-id.csv
mv uniprot_full_wo-valid-ood.csv.005 uniprot_full_valid-id.csv
```

Get another 0.5% subsample of the csv data file for id test data file:
```
gawk 'BEGIN {srand()} {f = FILENAME (rand() <= 0.995 ? ".995" : ".005"); print > f}' uniprot_full_wo-valid-ood-id.csv
mv uniprot_full_wo-valid-ood-id.csv.995 uniprot_full_wo-valid-ood-id-test-id.csv
mv uniprot_full_wo-valid-ood-id.csv.005 uniprot_full_test-id.csv
```

Get uniprot accn of valid and test id split:
```
cut -d, -f1,2 uniprot_full_valid-id.csv > uniprot_full_valid-id_accn.csv
cut -d, -f1,2 uniprot_full_test-id.csv > uniprot_full_test-id_accn.csv
```

Create offset dict jsons:
```
python preproc/create_offset_dict.py -i ../data/uniprot_full_valid-ood.csv -o ../data/uniprot_full_valid-ood_offsetdict.json;
python preproc/create_offset_dict.py -i ../data/uniprot_full_valid-id.csv -o ../data/uniprot_full_valid-id_offsetdict.json;
python preproc/create_offset_dict.py -i ../data/uniprot_full_test-id.csv -o ../data/uniprot_full_test-id_offsetdict.json;
(python preproc/create_offset_dict.py -i ../data/uniprot_full_wo-valid-ood-id.csv -o ../data/uniprot_full_wo-valid-ood-id_offsetdict.json;)
python preproc/create_offset_dict.py -i ../data/uniprot_full_wo-valid-ood-id-test-id.csv -o ../data/uniprot_full_wo-valid-ood-id-test-id_offsetdict.json;
```
