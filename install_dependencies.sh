if [[ ! -d "./ncbi-blast-2.13.0+" ]];
then
    if [[ ! -f "./ncbi-blast-2.13.0+-x64-linux.tar.gz" ]];
    then
        wget "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.13.0/ncbi-blast-2.13.0+-x64-linux.tar.gz"
    fi
    tar -xf "./ncbi-blast-2.13.0+-x64-linux.tar.gz"
fi

if [[ ! -d "blastdb/swissprot" ]];
then
    if [[ ! -f "./swissprot.tar.gz" ]];
    then
        wget "https://ftp.ncbi.nlm.nih.gov/blast/db/swissprot.tar.gz"
    fi
    mkdir -p blastdb/swissprot
    tar -xf "./swissprot.tar.gz" -C blastdb/swissprot
fi

if [[ ! -d "blastdb/pdbaa" ]];
then
    if [[ ! -f "./pdbaa.tar.gz" ]];
    then
        wget "https://ftp.ncbi.nlm.nih.gov/blast/db/pdbaa.tar.gz"
    fi
    mkdir -p blastdb/pdbaa
    tar -xf "./swissprot.tar.gz" -C blastdb/pdbaa
fi

mkdir -p temp
mkdir -p find_pdb
rm -f ./*.tar.gz

python3 -m pip install -r ./requirements.txt

echo "All done."


