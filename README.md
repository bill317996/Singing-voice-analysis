# Singing-voice-analysis
A pytorch model for singing-voice-analysis. (38 artists selected)

### Dependencies

Requires following packages:

- python 3.6
- pytorch 1.0.0
- numpy
- librosa

### Usage
Put your audio files in "./input/" folder and run
```
python main.py
```
or
```
python3 main.py
```
#### main.py
```
usage: main.py [-h] [-in IN_PATH] [-o OUT_PATH] [--cuda] [-gid GPU_INDEX]
optional arguments:
  -h, --help            show this help message and exit
  -in IN_PATH, --in_path IN_PATH
                        Path to input audios folder (default: ./input/)
  -o OUT_PATH, --out_path OUT_PATH
                        Path to output folder (default: ./output/)
  --cuda                use GPU computation
  -gid GPU_INDEX, --gpu_index GPU_INDEX
                        Assign a gpu index for processing if cuda. (default: 0)
```
#### Result format
```
emb_256.npy:  Latent space of singing voice. (256 dimension numpy array)
art_38:       Result of 38 class artist classification. (38 dimension numpy array)
sc_2:         Result of singing voice characteristic. (2 dimension numpy array)

The list correspond to result numpy array:
artist_list = [ '張惠妹', '郭靜', '蔡依林', '劉若英', '徐佳瑩', '田馥甄', '蔡健雅', '梁靜茹', '鄧紫棋', '孫燕姿',
                '費玉清', '張學友', '王力宏', '周杰倫', '陳奕迅', '林志炫', '林俊傑', '蕭敬騰', '盧廣仲', '李榮浩',
                '郭美美', '羅志祥', 'Amy Winehouse', '方大同', '王心凌', 'Erykah Badu', 'Macy Gray', 'Rihanna', '潘瑋柏', 
                '王若琳', 'Norah Jones', 'Pussycat Dolls', '畢書盡', '楊丞琳', '汪峰', '江美琪', 'Taylor Swift', '回聲樂團']
sc_list = ['negative', 'positive']
```
