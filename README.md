## Environment
* OS : Ubuntu 18.04
* GPU : GeForce RTX 3090 * 1
* CUDA : cuda 11.2
* PYTORCH : pytorch 1.7.1+cu110

## Training
Data Pre-setting

    git clone https://github.com/e9t/nsmc.git
    python datasetting.py
    cat ./data/nsmc_train.txt ./data/nsmc_test.txt > ./data/nsmc.txt
    cat ./data/corpus/nsmc_train.txt ./data/corpus/nsmc_test.txt > ./data/corpus/nsmc.txt

Data Preprocessing
    
    python remove_words.py # Tokenization, Remove Stopwords
    python build_graph.py # Generate Graph, Embedding
    python pre_dataloader.py --dataset nsmc # Data Padding, Preprocessing
    
Training

    python train.py --dataset nsmc

Check Accuracy with test dataset
    
    python test.py --dataset nsmc --test_epoch {test_epoch}
    
* If you want to use pretrained models, click [here](https://drive.google.com/drive/folders/16C3WE9KnpscdB7aQTvsRps7-OzjIhO8f?usp=sharing) and download pth files.
    
## Visualize

    # visualize.txt에 분류할 텍스트 입력
    python visualize.py
    
## Result

At epoch 50, 
- accuracy :  0.7825276575276575
- f1_score :  0.777611542639313
- recall_score :  0.7550646766169155

## Reference
* [불용어 출처](https://www.ranks.nl/stopwords/korean)
* [임베딩 모델](https://github.com/Kyubyong/wordvectors)
* [Pytorch Code Baseline](https://github.com/Niousha12/Text_Classification_via_GNN)
* [utils.py, remove_words.py](https://github.com/CRIPAC-DIG/TextING)

## Citation
    @inproceedings{zhang2020every,
    title={Every Document Owns Its Structure: Inductive Text Classification via Graph Neural Networks},
    author={Zhang, Yufeng and Yu, Xueli and Cui, Zeyu and Wu, Shu and Wen, Zhongzhen and Wang, Liang},
    booktitle="Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    year={2020}
    }
