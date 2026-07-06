# Multimodal Radiology System

Göğüs röntgeni görüntüleri ile klinik raporların **çok-kipli (multimodal) füzyonu** yoluyla,
sınıf dengesizliği altında **çok-etiketli göğüs hastalığı tespiti** yapan referans uygulama.

Model mimarisi (bkz. `proposed_method.py`):

- **Görüntü kolu:** EfficientNet-B0 (ImageNet ağırlıkları) → `Linear(·, 512)`
- **Metin kolu:** BERT `bert-base-uncased`, `pooler_output(768)` → `Linear(768, 512)`
- **Geç füzyon:** birleştirme → `R^1024` → `Linear(1024, 256)` → ReLU → `Linear(256, 14)`
- **Çıkış:** her bulgu için bağımsız Sigmoid (çok-etiketli)
- **Kayıp:** çok-etiketli Focal Loss (α = 0.5, γ = 2)
- **Karar eşiği:** τ = 0.30

---

## ⚠️ Kurulum — Önce `archive.zip` dosyasını açın

Veri kümeleri ve yardımcı kaynaklar boyutları nedeniyle depoya (git) dahil **edilmemiştir**;
bunun yerine **`archive.zip`** olarak ayrıca dağıtılır.

`proposed_method.py` çalışmadan önce bu arşivin proje kök dizininde açılmış olması gerekir.
Açıldığında proje köküne bir **`archive/`** klasörü oluşur ve kod bu klasördeki yolları kullanır.

### 1. Arşivi açın

Arşivi **proje kök dizinine** (`README.md` ve `proposed_method.py` ile aynı yere) açın:

```powershell
# Windows / PowerShell
Expand-Archive -Path archive.zip -DestinationPath . -Force
```

```bash
# Linux / macOS (veya Git Bash)
unzip archive.zip -d .
```

> Açma sonrası proje kökünde `archive/` klasörünü görmelisiniz.
> Zip içindeki üst klasör zaten `archive/` olduğu için hedef olarak `.` (proje kökü) verilir —
> ayrıca bir `archive/archive/...` iç içe klasörü oluşmamalıdır.

### 2. Bağımlılıkları kurun

```bash
pip install -r requirements.txt
# GPU eğitimi için torch/torchvision'ın CUDA sürümünü https://pytorch.org adresinden kurun.
```

### 3. Çalıştırın

```bash
python proposed_method.py
```

Betik önce eğitir, ardından ayrılmış test kümesinde değerlendirir. Yollar varsayılan olarak
`archive/` içindeki verilere göre ayarlıdır; farklı bir konum kullanacaksanız
`proposed_method.py` içindeki `CONFIG` bloğunda yalnızca `DATA_DIR` ve `IMG_DIR` değerlerini düzenleyin.

---

## Dizin Yapısı

`archive.zip` açıldıktan sonraki tam yapı aşağıdadır. `⭐` ile işaretli yollar
`proposed_method.py` tarafından doğrudan kullanılır.

```
multimodal-radiology-system/
├── proposed_method.py          # Ana model — eğitim + değerlendirme (tek dosyada)
├── requirements.txt            # Python bağımlılıkları
├── README.md
├── archive.zip                 # ⬇️ Veri + kaynak arşivi (çalıştırmadan önce açılmalı)
└── archive/                    # ── archive.zip açıldığında oluşur ──
    │
    ├── MedViLL-master/
    │   └── MedViLL-master/     # MedViLL kaynak reposu (referans/karşılaştırma)
    │       ├── data/
    │       │   ├── openi/      # ⭐ DATA_DIR
    │       │   │   ├── Train.jsonl   # eğitim: görüntü yolu + rapor + etiketler
    │       │   │   ├── Valid.jsonl
    │       │   │   └── Test.jsonl
    │       │   ├── mimic/
    │       │   └── vqa_rad/
    │       ├── CNN_BERT/
    │       ├── downstream_task/      # sınıflandırma / retrieval / vqa
    │       ├── models/
    │       ├── utils/
    │       └── main.py
    │
    ├── openi/
    │   └── home/data_storage/mimic-cxr/dataset/open_i/image_preprocessing/
    │       └── 512_3ch/        # ⭐ IMG_DIR — 3.547 göğüs röntgeni (512×512, 3 kanal)
    │
    ├── mimic_dset/
    │   └── re_512_3ch/         # MIMIC-CXR görüntüleri (512×512, 3 kanal)
    │       ├── Train/          # 89.607 jpg
    │       ├── Valid/          #    760 jpg
    │       └── Test/           #  1.541 jpg
    │
    ├── Cleanses csv tfrecords/ # temizlenmiş CSV + TFRecord'lar
    │   ├── df_train.csv, df_val.csv
    │   ├── df_train_partial_{5000,10000,30000}.csv
    │   └── CXR_train*.tfrecord, CXR_val.tfrecord
    │
    ├── NLP_aug_datasets/       # T5 ile artırılmış metin verisi
    │   ├── df_train_aug.csv, df_val_aug.csv
    │   └── T5_aug_CXR_train.tfrecord, T5_paired_CXR_val.tfrecord
    │
    └── shorter_summ_aug/       # kısaltılmış özet (maxlen=64) artırımı
        ├── maxlen64_T5_aug_CXR_train_50000.tfrecord
        └── maxlen64_T5_paired_CXR_val.tfrecord
```

Açılan `archive/` klasörü toplam ~8.5 GB'dir (çoğunluğu MIMIC-CXR görüntüleri).

## Notlar

- `archive/`, `archive.zip` ve eğitim çıktıları (`.pth`) `.gitignore` ile hariç tutulmuştur.
- `proposed_method.py` eğitim tamamlandığında en iyi kontrol noktasını
  `proposed_method.pth` olarak proje köküne kaydeder.
