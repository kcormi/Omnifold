for seed in {1..20}
  do
    echo seed ${seed}
    #sed -i "/--bsseed/c\ python\ mytrain.py\ -m\ omnifold\ -u\ omnifold\ --input-dim\ 3\ -e\ 50\ -ui\ 6\ --bootstrap\ --bsseed\ ${seed}" train_gpu.sh
    #sed -i "/--bsseed/c\ python\ mytrain.py\ -m\ multifold\ -u\ manyfold\ -e\ 50\ -ui\ 6\ --bootstrap\ --bsseed\ ${seed}" train_gpu.sh 
    #sed -i "/--bsseed/c\ python\ mytrain.py\ -m\ multifold\ -u\ manyfold\ -e\ 50\ -ui\ 6\ -data\ Pythia8CP1\ --bootstrap\ --bsseed\ ${seed}" train_gpu.sh 
    #sed -i "/--bsseed/c\ python\ mytrain.py\ -m\ omnifold\ -u\ omnifold\ --input-dim\ 3\ -e\ 50\ -ui\ 6\ -data\ Pythia8CP1\ --bootstrap\ --bsseed\ ${seed}" train_gpu.sh
    #sed -i "/--bsseed/c\ python\ mytrain.py\ -m\ omnifold\ -u\ omnifold\ --input-dim\ 3\ -e\ 50\ -ui\ 6\ --save-best-only\ --weight-clip-max\ 100.0\ --bootstrap\ --bsseed\ ${seed}" train_gpu.sh
    #sed -i "/--bsseed/c\ python\ mytrain.py\ -m\ unifold\ -u\ unifold\ -e\ 50\ -ui\ 6\ --save-best-only\ --weight-clip-max\ 100.0\ --bootstrap\ --bsseed\ ${seed}" train_gpu.sh
    #sed -i "/--MCbsseed/c\ python\ mytrain.py\ -m\ unifold\ -u\ unifold\ -data\ Pythia8CP1_tuneES\ -e\ 50\ -ui\ 6\ --save-best-only\ --weight-clip-max\ 100.0\ --MCbootstrap\ --MCbsseed\ ${seed}" train_gpu.sh
    #sed -i "/--bsseed/c\ python\ mytrain.py\ -m\ omnifold\ -u\ omnifold\ -data\ Pythia8CP1_tuneES\ --input-dim\ 3\ -e\ 50\ -ui\ 6\ --save-best-only\ --weight-clip-max\ 100.0\ --bootstrap\ --bsseed\ ${seed}" train_gpu.sh
    #sed -i "/--bsseed/c\ python\ mytrain.py\ -m\ multifold\ -u\ manyfold\ -data\ Pythia8CP1_tuneES\ -e\ 50\ -ui\ 6\ --save-best-only\ --weight-clip-max\ 100.0\ --bootstrap\ --bsseed\ ${seed}" train_gpu.sh
    #sed -i "/--MCbsseed/c\ python\ mytrain.py\ -m\ omnifold\ -u\ omnifold\ -data\ Pythia8CP1_tuneES\ --input-dim\ 3\ -e\ 50\ -ui\ 6\ --save-best-only\ --weight-clip-max\ 100.0\ --MCbootstrap\ --MCbsseed\ ${seed}" train_gpu.sh
    #sed -i "/--MCbsseed/c\ python\ mytrain.py\ -m\ multifold\ -u\ manyfold\ -data\ Pythia8CP1_tuneES\ -e\ 50\ -ui\ 6\ --save-best-only\ --weight-clip-max\ 100.0\ --MCbootstrap\ --MCbsseed\ ${seed}" train_gpu.sh

    #sed -i "/--bsseed/c\ python\ mytrain.py\ -m\ omnifold\ -u\ omnifold\ -mc\ Pythia8CP1_tuneES\ --input-dim\ 3\ -e\ 50\ -ui\ 6\ --weight-clip-max\ 100.0\ --save-best-only\ --bootstrap\ --bsseed\ ${seed}" train_gpu.sh
    #sed -i "/--bsseed/c\ python\ mytrain.py\ -m\ multifold\ -u\ manyfold\ -mc\ Pythia8CP1_tuneES\ -e\ 50\ -ui\ 8\ --weight-clip-max\ 100.0\ --save-best-only\ --bootstrap\ --bsseed\ ${seed}" train_gpu.sh
    #sed -i "/--bsseed/c\ python\ mytrain.py\ -m\ unifold\ -u\ unifold\ -mc\ Pythia8CP1_tuneES\ -e\ 50\ -ui\ 6\ --weight-clip-max 100.0\ --save-best-only\ --bootstrap\ --bsseed\ ${seed}" train_gpu.sh
    #sed -i "/--bsseed/c\ python\ mytrain.py\ -m\ omnifold\ -u\ omnifold\ -mc\ Pythia8CP1_tuneES\ -data\ Pythia8CP5\ --input-dim\ 3\ -e\ 50\ -ui\ 6\ --weight-clip-max\ 100.0\ --save-best-only\ --bootstrap\ --bsseed\ ${seed}" train_gpu.sh
    #sed -i "/--bsseed/c\ python\ mytrain.py\ -m\ multifold\ -u\ manyfold\ -mc\ Pythia8CP1_tuneES\ -data\ Pythia8CP5\ -e\ 50\ -ui\ 6\ --weight-clip-max\ 100.0\ --save-best-only\ --bootstrap\ --bsseed\ ${seed}" train_gpu.sh
#    sed -i "/--bsseed/c\ python\ mytrain.py\ -m\ unifold\ -u\ unifold\ -mc\ Pythia8CP1_tuneES\ -data\ Pythia8CP5\ -e\ 50\ -ui\ 6\ --weight-clip-max 100.0\ --save-best-only\ --bootstrap\ --bsseed\ ${seed}" train_gpu.sh

    #sed -i "/--MCbsseed/c\ python\ mytrain.py\ -m\ omnifold\ -u\ omnifold\ -mc\ Pythia8CP1_tuneES\ --input-dim\ 3\ -e\ 50\ -ui\ 6\ --weight-clip-max\ 100.0\ --save-best-only\ --MCbootstrap\ --MCbsseed\ ${seed}" train_gpu.sh
    #sed -i "/--MCbsseed/c\ python\ mytrain.py\ -m\ multifold\ -u\ manyfold\ -mc\ Pythia8CP1_tuneES\ -e\ 50\ -ui\ 6\ --save-best-only\ --weight-clip-max\ 100.0\ --MCbootstrap\ --MCbsseed\ ${seed}" train_gpu.sh
    #sed -i "/--MCbsseed/c\ python\ mytrain.py\ -m\ unifold\ -u\ unifold\ -mc\ Pythia8CP1_tuneES\ -e\ 50\ -ui\ 6\ --weight-clip-max 100.0\ --save-best-only\ --MCbootstrap\ --MCbsseed\ ${seed}" train_gpu.sh

    #sed -i "/--bsseed/c\ python\ mytrain.py\ -m\ multifold\ -u\ manyfold\ -mc\ Pythia8EPOS\ -e\ 50\ -ui\ 8\ --weight-clip-max\ 100.0\ --save-best-only\ --bootstrap\ --bsseed\ ${seed}" train_gpu.sh

    #sed -i "/--MCbsseed/c\ python\ mytrain.py\ -m\ omnifold\ -u\ omnifold\ -mc\ Pythia8EPOS\ --input-dim\ 3\ -e\ 50\ -ui\ 6\ --weight-clip-max\ 100.0\ --save-best-only\ --MCbootstrap\ --MCbsseed\ ${seed}" train_gpu.sh
    #sed -i "/--MCbsseed/c\ python\ mytrain.py\ -m\ multifold\ -u\ manyfold\ -mc\ Pythia8EPOS\ -e\ 50\ -ui\ 6\ --save-best-only\ --weight-clip-max\ 100.0\ --MCbootstrap\ --MCbsseed\ ${seed}" train_gpu.sh

    #sed -i "/--bsseed/c\ python\ mytrain.py\ -m\ omnifold\ -u\ omnifold\ -mc\ Pythia8EPOS\ --input-dim\ 3\ -e\ 50\ -ui\ 6\ --weight-clip-max\ 100.0\ --save-best-only\ --bootstrap\ --bsseed\ ${seed}" train_gpu.sh
    #sed -i "/--bsseed/c\ python\ mytrain.py\ -m\ multifold\ -u\ manyfold\ -mc\ Pythia8EPOS\ -e\ 50\ -ui\ 6\ --save-best-only\ --weight-clip-max\ 100.0\ --bootstrap\ --bsseed\ ${seed}" train_gpu.sh

    sed -i "/--bsseed/c\ python\ mytrain.py\ -m\ omnifold\ -u\ omnifold\ -mc\ Pythia8EPOS\ -data Pythia8CP5 --input-dim\ 3\ -e\ 50\ -ui\ 6\ --weight-clip-max\ 100.0\ --save-best-only\ --bootstrap\ --bsseed\ ${seed}" train_gpu.sh

    sbatch train_gpu.sh
  done

