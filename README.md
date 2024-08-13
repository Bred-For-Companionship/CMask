# Code for *Cirruculum Masking to Maximize Cross Modal Interaction in Vision and Language Pretraining* (NAACL '24)

[![Paper](https://img.shields.io/badge/NAACL-2024-blue)](https://aclanthology.org/2024.naacl-long.203/)
[![Email Badge](https://img.shields.io/badge/Gmail-Contact_Me-green?style=flat-square&logo=gmail&logoColor=FFFFFF&labelColor=3A3B3C&color=62F1CD)](mailto:ytou3@gatech.edu)

![Model Architecture](https://github.com/Bred-For-Companionship/CMask/blob/main/dataset/model_architecture.png)

### TODO

- [x] Support parallel training of agent and main model
- [ ] Upload interpretability experiments
- [ ] Support bridging to encoder-decoder VLMs
- [ ] Document inference code usage

## 📚 Citation
```
 @inproceedings{tou2024curriculum,
  title={Curriculum Masking in Vision-Language Pretraining to Maximize Cross Modal Interaction},
  author={Tou, Kraig and Sun, Zijun},
  booktitle={Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={3672--3688},
  year={2024}
}
```
### Usage

**Download data (if not downloaded)**
To use auto download, first create an environment for LLAVIS
```
conda create -n lavis python=3.8
conda activate lavis
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e .

```
Then change the download directory to the data folder for CMASK:
```
sed -i 's|cache_root: "/export/home/.cache/lavis"|cache_root: "../CMASK/data"|' lavis/configs/default.yaml
```
Afterwards, download the necessary pretrain data (or use custom ones) via (for example, for coco):
```
python download_coco.py
```
Now the conda environment for LLavis can be deleted and a new one for CMASK may be created to install requirements.txt

**Pretrain**
Change nproc_per_node to indicate number of GPU's and output directory as needed:

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env Pretrain.py --config ./configs/Pretrain.yaml --output_dir output/Pretrain
```


