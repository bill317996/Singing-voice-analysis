
import os
import torch
import numpy as np
import librosa
import argparse
import model

artist_list = ['張惠妹', '郭靜', '蔡依林', '劉若英', '徐佳瑩', '田馥甄', '蔡健雅', '梁靜茹', '鄧紫棋', '孫燕姿',
                '費玉清', '張學友', '王力宏', '周杰倫', '陳奕迅', '林志炫', '林俊傑', '蕭敬騰', '盧廣仲', '李榮浩',
                '郭美美', '羅志祥', 'Amy Winehouse', '方大同', '王心凌', 'Erykah Badu', 'Macy Gray', 'Rihanna', '潘瑋柏', 
                '王若琳', 'Norah Jones', 'Pussycat Dolls', '畢書盡', '楊丞琳', '汪峰', '江美琪', 'Taylor Swift', '回聲樂團']

def batchize(raw_stft, seg_len=430):
    """
    Divided entire song into several segments every 10 seconds (430 frames).
    """
    total_len = raw_stft.shape[1]
    seg_num = int(total_len / seg_len)
    for i in range(seg_num):
        if i == 0:
            data = raw_stft[None, :, :seg_len]
        else:
            data = np.concatenate((data, raw_stft[None, :, i*seg_len:(i+1)*seg_len]), axis=0)
    if total_len % seg_len != 0:
        tmp = np.zeros((raw_stft.shape[0], seg_len))
        tmp[:, :total_len%seg_len] = raw_stft[:, seg_num*seg_len:]
        data = np.concatenate((data, tmp[None, :, :]), axis=0)
    return data

def main(datadir, savedir, cuda, gid):

    print('===============')
    print('Singing Voice Analysis')
    print('Author: Bill Hsieh')
    print('Update in 20190123')
    print('===============')

    """
    Load pretrain model
    """

    pretrain_path = './model_state_dict'
    if cuda:
        pretrain_model = torch.load(pretrain_path, map_location={'cuda:1':'cuda:{}'.format(gid)})
    else:
        pretrain_model = torch.load(pretrain_path, map_location=lambda storage, loc: storage)

    Encoder = model.Encoder()
    if cuda:
        Encoder.cuda()
    Encoder.float()
    Encoder.load_state_dict(pretrain_model['Encoder_state_dict'])
    Encoder.eval()
    for p in Encoder.parameters():
        p.requires_grad = False

    NetD = model.NetD()
    NetD.float()
    if cuda:
        NetD.cuda()
    NetD.load_state_dict(pretrain_model['NetD_state_dict'])
    NetD.eval()
    for p in NetD.parameters():
        p.requires_grad = False

    NetC_art38 = model.NetC(38)
    NetC_art38.float()
    if cuda:    
        NetC_art38.cuda()
    NetC_art38.load_state_dict(pretrain_model['NetC_art38_state_dict'])
    NetC_art38.eval()
    for p in NetC_art38.parameters():
        p.requires_grad = False

    NetS = model.NetC(2)
    NetS.float()
    if cuda:
        NetS.cuda()
    NetS.load_state_dict(pretrain_model['NetS_state_dict'])
    NetS.eval()
    for p in NetS.parameters():
        p.requires_grad = False

    """
    audio process
    """

    for root, dirr, file in os.walk(datadir):
        dirr.sort()
        file.sort()
        for filename in file:
            if '.wav' in filename or '.mp3' in filename:

                songname = filename.split('.')[0]
                print('Songname: ', songname)

                """
                Use librosa to extract stft feature
                """
                
                fp = os.path.join(root, filename)
                sigs, sr = librosa.load(fp, sr=44100, mono=True)
                raw_stft = np.abs(librosa.stft(sigs, n_fft=2048, hop_length=1024))

                """
                Pre-process before prediction
                """
                
                batch_x = batchize(raw_stft)
                batch_x = torch.from_numpy(batch_x).float()
                if cuda:
                    batch_x = batch_x.cuda()
                batch_x = torch.log1p(batch_x)

                """
                Made prediction by model
                """
                
                osize, x1, x2, x = Encoder(batch_x)
                ss = NetD(x) # singers space
                pred_SID = NetC_art38(ss) # singer ID
                pred_sc = NetS(ss) # singer characteristic

                """
                torch to numpy
                """
                
                x = x.detach().cpu().numpy()
                ss = ss.detach().cpu().numpy()
                pred_SID = pred_SID.detach().cpu().numpy()
                pred_sc = pred_sc.detach().cpu().numpy()

                """
                Weighted average for each segments
                """
                
                weight = np.sum(np.sum(x, axis=1), axis=1)
                sweight = np.sum(weight)

                pred_wss = np.zeros(256)
                pred_wSID = np.zeros(38)
                pred_wsc = np.zeros(2)

                for i in range(len(weight)):
                    pred_wss += weight[i] * ss[i] 
                    pred_wSID += weight[i] * pred_SID[i] 
                    pred_wsc += weight[i] * pred_sc[i] 
                pred_wss = pred_wss / sweight
                pred_wSID = pred_wSID / sweight
                pred_wsc = pred_wsc / sweight

                """
                Save result
                """
                
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                np.savez(savedir+'/'+songname, emb_256=pred_wss, art_38=pred_wSID, sc_2=pred_wsc)
#                 np.save(savedir+'/'+songname+'/emb_256.npy', pred_wss)
#                 np.save(savedir+'/'+songname+'/art_38.npy', pred_wSID)
#                 np.save(savedir+'/'+songname+'/sc_2.npy', pred_wsc)

                """
                Print result
                """
                
                argsort_wSID = np.argsort(pred_wSID*(-1))
                print('*******')
                print('Artist Similarity: ')
                for item in argsort_wSID:
                    if pred_wSID[item] > 0.01:
                        print(artist_list[item], ': %.2f'% (pred_wSID[item]*100), '%')
                print('*******')
                print('Characteristic Ratio: %.2f'% (pred_wsc[1]*100), '%')
                print('===============')

def parser():
    
    p = argparse.ArgumentParser()

    p.add_argument('-in', '--in_path',
                    help='Path to input audios folder (default: %(default)s',
                    type=str, default='./input/')
    p.add_argument('-o', '--out_path',
                    help='Path to output folder (default: %(default)s',
                    type=str, default='./output/')

    p.add_argument('--cuda', action='store_true', 
                    help='use GPU computation')
    p.add_argument('-gid', '--gpu_index',
                    help='Assign a gpu index for processing if cuda. (default: %(default)s',
                    type=int, default=0)
    
    return p.parse_args()

if __name__ == '__main__':
    args = parser()
    if args.cuda:
        with torch.cuda.device(args.gpu_index):
            main(args.in_path, args.out_path, args.cuda, args.gpu_index)
    else:
        main(args.in_path, args.out_path, args.cuda, args.gpu_index)

