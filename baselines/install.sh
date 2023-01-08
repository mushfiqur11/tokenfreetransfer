#!/bin/bash
task=${task:-none}
lang=${lang:-eng}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        echo $1 $2 #Optional to see the parameter:value result
   fi

  shift
done

echo ${task}
echo ${lang}

module load python/3.8.6-ff
# cd /scratch/ffaisal/





if [[ "$task" = "install_adapter" || "$task" = "all" ]]; then
   echo "------------------------------Install adapter latest------------------------------"
   module load python/3.8.6-ff
   rm -rf adapter-transformers-l
   rm -rf vnv/vnv-adp-l
   python -m venv vnv/vnv-adp-l
   source vnv/vnv-adp-l/bin/activate
   wget -O adapters3.1.0.tar.gz https://github.com/adapter-hub/adapter-transformers/archive/refs/tags/adapters3.1.0.tar.gz
   tar -xf adapters3.1.0.tar.gz
   rm adapters3.1.0.tar.gz
   mv adapter-transformers-adapters3.1.0 adapter-transformers-l
   cd adapter-transformers-l
   #cp ../scripts/ad_l_trans_trainer.py src/transformers/trainer.py
   pip install adapter-transformers
   ../vnv/vnv-adp-l/bin/python -m pip install --upgrade pip
   cd ..
   pip install --upgrade pip
   pip3 install -r requirements.txt
   ##for A100 gpu
   deactivate
fi

if [[ "$task" = "install_transformer" || "$task" = "all" ]]; then
   echo "------------------------------Install transformer latest------------------------------"
   module load python/3.8.6-ff

   rm -rf transformers-orig
   rm -rf vnv/vnv-trns
   module load python/3.8.6-ff
   python -m venv vnv/vnv-trns
   source vnv/vnv-trns/bin/activate
   wget -O transformersv4.21.1.tar.gz "https://github.com/huggingface/transformers/archive/refs/tags/v4.21.1.tar.gz"
   tar -xf transformersv4.21.1.tar.gz
   rm transformersv4.21.1.tar.gz
   mv transformers-4.21.1 transformers-orig
   cd transformers-orig
   pip install .
   pip install --upgrade pip
   cd ..
   pip install -r requirements.txt
   deactivate
   mkdir tr_output
fi

if [[ "$task" = "download_data" || "$task" = "all" ]]; then
   echo "------------------------------Download mlm all train data-------------------------------"
   wget -O ud-treebanks-v2.11.tgz https://gmuedu-my.sharepoint.com/:u:/g/personal/ffaisal_gmu_edu/EWT62asjHfdKiQ0SQRRvvGEBBS5STS72eiqLEwuow3O9PQ?download=1
   tar zxvf ud-treebanks-v2.11.tgz
   rm ud-treebanks-v2.11.tgz
   rm -rf __MACOSX
   
fi

if [[ "$task" = "utils" || "$task" = "all" ]]; then
   rm vnv/vnv-adp-l/lib/python3.8/site-packages/transformers/trainer.py
   cp scripts/trainer.py vnv/vnv-adp-l/lib/python3.8/site-packages/transformers/
fi

if [[ "$task" = "train_udp" ]]; then
   export ALL_MODELS=("english-ewt" "arabic-padt" "coptic-scriptorium" "hindi-hdtb" "japanese-gsd" "korean-gsd" "tamil-ttb" "vietnamese-vtb" "chinese-gsd")
   # export ALL_MODELS=("english-ewt")

   for MODEL_NAME in ${ALL_MODELS[@]}; do
      efile="tr_output/${MODEL_NAME}_udp_train.err"
      ofile="tr_output/${MODEL_NAME}_udp_train.out"
      echo ${efile}
      echo ${ofile}
      sbatch -o ${ofile} -e ${efile} slurm/train_udp.slurm ${MODEL_NAME}
   done
fi


if [[ "$task" = "train_pos" ]]; then
   export ALL_MODELS=("english-ewt" "arabic-padt" "coptic-scriptorium" "hindi-hdtb" "japanese-gsd" "korean-gsd" "tamil-ttb" "vietnamese-vtb" "chinese-gsd")
   # export ALL_MODELS=("english-ewt")

   for MODEL_NAME in ${ALL_MODELS[@]}; do
      efile="tr_output/${MODEL_NAME}_pos_train.err"
      ofile="tr_output/${MODEL_NAME}_pos_train.out"
      echo ${efile}
      echo ${ofile}
      sbatch -o ${ofile} -e ${efile} slurm/train_pos.slurm ${MODEL_NAME}
   done
fi

if [[ "$task" = "train_ner" ]]; then
   export ALL_MODELS=("en" "ar" "hi" "ja" "ko" "ta" "vi" "zh")
   # export ALL_MODELS=("english-ewt")

   for MODEL_NAME in ${ALL_MODELS[@]}; do
      efile="tr_output/${MODEL_NAME}_ner_train.err"
      ofile="tr_output/${MODEL_NAME}_ner_train.out"
      echo ${efile}
      echo ${ofile}
      sbatch -o ${ofile} -e ${efile} slurm/train_ner.slurm ${MODEL_NAME}
   done
fi

if [[ "$task" = "evaluate_udp1" ]]; then
   export ALL_MODELS=("english-ewt" "arabic-padt" "coptic-scriptorium" "hindi-hdtb" "japanese-gsd" "korean-gsd" "tamil-ttb" "vietnamese-vtb" "chinese-gsd")
   # export ALL_MODELS=("english-ewt")

   for MODEL_NAME in ${ALL_MODELS[@]}; do
      efile="tr_output/${MODEL_NAME}_udp_eval1.err"
      ofile="tr_output/${MODEL_NAME}_udp_eval1.out"
      echo ${efile}
      echo ${ofile}
      sbatch -o ${ofile} -e ${efile} slurm/eval_udp1.slurm ${MODEL_NAME} ${task}
   done
fi

if [[ "$task" = "evaluate_pos1" ]]; then
   export ALL_MODELS=("english-ewt" "arabic-padt" "coptic-scriptorium" "hindi-hdtb" "japanese-gsd" "korean-gsd" "tamil-ttb" "vietnamese-vtb" "chinese-gsd")
   export ALL_MODELS=("english-ewt")

   for MODEL_NAME in ${ALL_MODELS[@]}; do
      efile="tr_output/${MODEL_NAME}_pos_eval1.err"
      ofile="tr_output/${MODEL_NAME}_pos_eval1.out"
      echo ${efile}
      echo ${ofile}
      sbatch -o ${ofile} -e ${efile} slurm/eval_pos1.slurm ${MODEL_NAME} ${task}
   done
fi

if [[ "$task" = "evaluate_ner1" ]]; then
   export ALL_MODELS=("en" "ar" "hi" "ja" "ko" "ta" "vi" "zh")
   # export ALL_MODELS=("english-ewt")

   for MODEL_NAME in ${ALL_MODELS[@]}; do
      efile="tr_output/${MODEL_NAME}_ner_eval1.err"
      ofile="tr_output/${MODEL_NAME}_ner_eval1.out"
      echo ${efile}
      echo ${ofile}
      sbatch -o ${ofile} -e ${efile} slurm/eval_ner1.slurm ${MODEL_NAME} ${task}
   done
fi

if [[ "$task" = "evaluate_ner2" ]]; then
   export ALL_MODELS=("en" "ar" "hi" "ja" "ko" "ta" "vi" "zh")
   export ALL_PRED_MODELS=("ace" "af" "am" "ar" "as" "ast" "ay" "az" "ba" "bar" "be" "bg" "bn" "bo" "br" "ca" "ce" "ceb" "crh" "cs" "csb" "cv" "cy" "da" "de" "dv" "el" "en" "es" "et" "eu" "fa" "fi" "fo" "fr" "frr" "fy" "ga" "gd" "gl" "gn" "gu" "he" "hi" "hsb" "hu" "hy" "ig" "ilo" "is" "it" "ja" "jv" "ka" "kk" "km" "kn" "ko" "ksh" "ku" "ky" "lb" "lij" "ln" "lt" "lv" "mg" "mi" "mk" "ml" "mn" "mr" "ms" "mt" "my" "mzn" "nap" "nds" "ne" "nl" "no" "oc" "or" "os" "pa" "pl" "pms" "ps" "pt" "qu" "rm" "ro" "ru" "rw" "sah" "sd" "sh" "si" "sk" "sl" "so" "sq" "su" "sv" "sw" "ta" "te" "tg" "th" "tk" "tl" "tr" "tt" "ug" "uk" "ur" "uz" "vep" "vi" "war" "xmf" "yi" "yo" "zea" "zh")

   # export ALL_MODELS=("en")
   # export ALL_PRED_MODELS=("ace" "af")
   # export ALL_MODELS=("english-ewt")

   for MODEL_NAME in ${ALL_MODELS[@]}; do
      for MODEL_PRED in ${ALL_PRED_MODELS[@]}; do
         efile="tr_output/${MODEL_NAME}_${MODEL_PRED}_ner_eval2.err"
         ofile="tr_output/${MODEL_NAME}_${MODEL_PRED}_ner_eval2.out"
         echo ${efile}
         echo ${ofile}
         sbatch -o ${ofile} -e ${efile} slurm/eval_ner1.slurm ${MODEL_NAME} ${task} ${MODEL_PRED}
      done
   done
fi


if [[ "$task" = "evaluate_pos2" ]]; then
   export ALL_MODELS=("english-ewt" "arabic-padt" "coptic-scriptorium" "hindi-hdtb" "japanese-gsd" "korean-gsd" "tamil-ttb" "vietnamese-vtb" "chinese-gsd")
   export ALL_PRED_MODELS=("Afrikaans-AfriBooms" "Akkadian-RIAO" "Amharic-ATT" "Ancient_Greek-Perseus" "Ancient_Hebrew-PTNK" "Arabic-PADT" "Armenian-ArmTDP" "Western_Armenian-ArmTDP" "Bambara-CRB" "Basque-BDT" "Belarusian-HSE" "Breton-KEB" "Bulgarian-BTB" "Buryat-BDT" "Cantonese-HK" "Catalan-AnCora" "Chinese-GSD" "Classical_Chinese-Kyoto" "Coptic-Scriptorium" "Croatian-SET" "Czech-CAC" "Danish-DDT" "Dutch-LassySmall" "English-EWT" "Erzya-JR" "Estonian-EWT" "Faroese-OFT" "Finnish-FTB" "French-GSD" "Galician-CTG" "German-GSD" "Gothic-PROIEL" "Greek-GDT" "Hebrew-HTB" "Hindi-HDTB" "Hungarian-Szeged" "Icelandic-Modern" "Indonesian-GSD" "Irish-IDT" "Italian-ISDT" "Japanese-GSD" "Kazakh-KTB" "Kiche-IU" "Komi_Zyrian-Lattice" "Korean-GSD" "Kurmanji-MG" "Latin-PROIEL" "Latvian-LVTB" "Lithuanian-HSE" "Maltese-MUDT" "Manx-Cadhan" "Mbya_Guarani-Dooley" "Naija-NSC" "North_Sami-Giella" "Norwegian-Nynorsk" "Old_Church_Slavonic-PROIEL" "Old_East_Slavic-TOROT" "Old_French-SRCMF" "Persian-Seraji" "Polish-PDB" "Pomak-Philotis" "Portuguese-GSD" "Romanian-ArT" "Russian-GSD" "Sanskrit-Vedic" "Scottish_Gaelic-ARCOSG" "Serbian-SET" "Slovak-SNK" "Slovenian-SSJ" "Spanish-GSD" "Swedish-LinES" "Tamil-TTB" "Thai-PUD" "Turkish-Kenet" "Turkish_German-SAGT" "Ukrainian-IU" "Upper_Sorbian-UFAL" "Urdu-UDTB" "Uyghur-UDT" "Vietnamese-VTB" "Welsh-CCG" "Western_Armenian-ArmTDP" "Wolof-WTB" "Xibe-XDT")

   # export ALL_MODELS=("english-ewt")
   # export ALL_PRED_MODELS=("Afrikaans-AfriBooms" "Akkadian-RIAO")

   for MODEL_NAME in ${ALL_MODELS[@]}; do
      for MODEL_PRED in ${ALL_PRED_MODELS[@]}; do
         efile="tr_output/${MODEL_NAME}_${MODEL_PRED}_pos_eval2.err"
         ofile="tr_output/${MODEL_NAME}_${MODEL_PRED}_pos_eval2.out"
         echo ${efile}
         echo ${ofile}
         sbatch -o ${ofile} -e ${efile} slurm/eval_pos1.slurm ${MODEL_NAME} ${task} ${MODEL_PRED}
      done
   done
fi

if [[ "$task" = "evaluate_udp2" ]]; then
   export ALL_MODELS=("english-ewt" "arabic-padt" "coptic-scriptorium" "hindi-hdtb" "japanese-gsd" "korean-gsd" "tamil-ttb" "vietnamese-vtb" "chinese-gsd")
   export ALL_PRED_MODELS=("Afrikaans-AfriBooms" "Akkadian-RIAO" "Amharic-ATT" "Ancient_Greek-Perseus" "Ancient_Hebrew-PTNK" "Arabic-PADT" "Armenian-ArmTDP" "Western_Armenian-ArmTDP" "Bambara-CRB" "Basque-BDT" "Belarusian-HSE" "Breton-KEB" "Bulgarian-BTB" "Buryat-BDT" "Cantonese-HK" "Catalan-AnCora" "Chinese-GSD" "Classical_Chinese-Kyoto" "Coptic-Scriptorium" "Croatian-SET" "Czech-CAC" "Danish-DDT" "Dutch-LassySmall" "English-EWT" "Erzya-JR" "Estonian-EWT" "Faroese-OFT" "Finnish-FTB" "French-GSD" "Galician-CTG" "German-GSD" "Gothic-PROIEL" "Greek-GDT" "Hebrew-HTB" "Hindi-HDTB" "Hungarian-Szeged" "Icelandic-Modern" "Indonesian-GSD" "Irish-IDT" "Italian-ISDT" "Japanese-GSD" "Kazakh-KTB" "Kiche-IU" "Komi_Zyrian-Lattice" "Korean-GSD" "Kurmanji-MG" "Latin-PROIEL" "Latvian-LVTB" "Lithuanian-HSE" "Maltese-MUDT" "Manx-Cadhan" "Mbya_Guarani-Dooley" "Naija-NSC" "North_Sami-Giella" "Norwegian-Nynorsk" "Old_Church_Slavonic-PROIEL" "Old_East_Slavic-TOROT" "Old_French-SRCMF" "Persian-Seraji" "Polish-PDB" "Pomak-Philotis" "Portuguese-GSD" "Romanian-ArT" "Russian-GSD" "Sanskrit-Vedic" "Scottish_Gaelic-ARCOSG" "Serbian-SET" "Slovak-SNK" "Slovenian-SSJ" "Spanish-GSD" "Swedish-LinES" "Tamil-TTB" "Thai-PUD" "Turkish-Kenet" "Turkish_German-SAGT" "Ukrainian-IU" "Upper_Sorbian-UFAL" "Urdu-UDTB" "Uyghur-UDT" "Vietnamese-VTB" "Welsh-CCG" "Western_Armenian-ArmTDP" "Wolof-WTB" "Xibe-XDT")

   # export ALL_MODELS=("english-ewt")
   # export ALL_PRED_MODELS=("Afrikaans-AfriBooms" "Akkadian-RIAO")

   for MODEL_NAME in ${ALL_MODELS[@]}; do
      for MODEL_PRED in ${ALL_PRED_MODELS[@]}; do
         efile="tr_output/${MODEL_NAME}_${MODEL_PRED}_udp_eval2.err"
         ofile="tr_output/${MODEL_NAME}_${MODEL_PRED}_udp_eval2.out"
         echo ${efile}
         echo ${ofile}
         sbatch -o ${ofile} -e ${efile} slurm/eval_udp1.slurm ${MODEL_NAME} ${task} ${MODEL_PRED}
      done
   done
fi




