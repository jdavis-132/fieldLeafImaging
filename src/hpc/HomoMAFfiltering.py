from collections import Counter
import sys
from tqdm import tqdm


pathtovcf=sys.argv[1]
pathtofilteredvcf=sys.argv[2]

data=open(str(pathtovcf))
with open(str(pathtofilteredvcf), 'w') as SNPs:
    for line in tqdm(data):
        line=line.strip()
        if line.startswith("#"):
            # print(line)
            SNPs.write(f'{line}\n')
            continue

        fields=line.split("\t")
        # print(fields)

        pos=fields[2]

        genotypes=fields[9:]

        genotype_counts = Counter(genotypes)
        hom_ref = genotype_counts.get('0/0',0) + genotype_counts.get('0|0',0)
        hom_alt = genotype_counts.get('1/1',0) + genotype_counts.get('1|1',0)
        het_count = genotype_counts.get('0/1',0) + genotype_counts.get('0|1',0) +genotype_counts.get('1/0',0) + genotype_counts.get('1|0',0)
        total_count=len(genotypes)

        hom_maf = min(hom_ref, hom_alt) / total_count
        het_f = het_count/total_count

        if (het_f < 0.5 * hom_maf) and hom_maf >= 0.05:
                SNPs.write(f'{line}\n')
