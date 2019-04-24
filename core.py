import os
from flask import Flask, Response, request, abort, send_from_directory, render_template
from PIL import Image
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.signal import get_window
import os, sys
import cv2
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'library/'))
import stft
import energy
import time
import pyaudio

app = Flask(__name__)
_author__ = 'May Divine'
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}

pl = open('models/db_padalingsa/padalingsa.txt', 'r')
judul = []
padalingsa = []
for p in pl:
    p = p.strip()
    judul.append(p.split(':')[0])
    padalingsa.append(p.split(':')[1].split(','))

models_path = open('model path.txt', 'r')
model_path = []
for mp in models_path:
    mp = mp.strip()
    model_path.append(mp)
models = np.array([])
for path in model_path:
    models = np.append(models, [os.path.join(path,fname) for fname in os.listdir(path) if fname.endswith('.png')])
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def brute_force(strings):
    result = []
    matches = []
    matches_judul = []
    defined = False
    for p in range(len(padalingsa)):
        if len(padalingsa[p]) == len(strings):
            defined = True
            match = []
            for s in range(len(strings)):
                match.append(char_comparison(strings[s], padalingsa[p][s]))
            matches.append(np.sum(match))
            matches_judul.append(judul[p])
    if defined == False:
        return ['undefined']
    result.append(matches_judul[np.argmin(matches)])
    return result
	
def char_comparison(char1, char2):
    characters = ['a', 'e', 'i', 'o', 'u']
    for i in range(len(characters)):
        if char1 == characters[i]:
            enum_ch1 = i
        if char2 == characters[i]:
            enum_ch2 = i
    return abs(enum_ch1 - enum_ch2)

def create_spectrogram(mX, H, rates, N, filename, i):
    maxplotfreq = 5000.0
    numFrames = int(mX[:,0].size)
    frmTime = H*np.arange(numFrames)/float(rates)
    binFreq = rates*np.arange(N*maxplotfreq/rates)/N
    plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:int(N*maxplotfreq/rates+1)]))
    plt.axis("off")
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0, wspace = 0)
    plt.savefig("models/test/"+filename+"_"+str(i)+".png")
    plt.close()
    
def detect_silence(rates, M, Energy):
    Eaverage = np.average(Energy)
    sMinDuration = 0.8      #Minimum duration to identified as silence
    sMin = int(rates/float(M)*sMinDuration)     #sMinDuration as samples
    Etconst = 0.25 #0.124
    Ethreshold = Etconst * Eaverage
    Esilences = np.array([], dtype=np.int32)
    counter = []
    for x in range(Energy.size):
        if Energy[x] < Ethreshold:
            counter.append(x)
            if len(counter) >= sMin and x == Energy.size - 1:
                Esilences = np.append(Esilences, [counter[0], counter[-1]])
        else:
            if len(counter) >= sMin:
                Esilences = np.append(Esilences, [counter[0], counter[-1]])
            counter = []
    Esilences = np.reshape(Esilences,(-1,2))
    return Esilences

def readImage(imgpath, WIDTH, HEIGHT):
    im = Image.open(imgpath)
    w, h = im.size
    aspect = 1.0*w/h
    if aspect > 1.0*WIDTH/HEIGHT:
        width = min(w, WIDTH)
        height = width/aspect
    else:
        height = min(h, HEIGHT)
        width = height*aspect
    return {'width': int(width), 'height': int(height),'src': imgpath}

@app.route('/<path:filename>')
def image(filename):
    try:
        w = int(request.args['w'])
        h = int(request.args['h'])
    except (KeyError, ValueError):
        return send_from_directory('.', filename)

    try:
        im = Image.open(filename)
        im.thumbnail((w, h), Image.ANTIALIAS)
        io = StringIO.StringIO()
        im.save(io, format='JPEG')
        return Response(io.getvalue(), mimetype='image/jpeg')

    except IOError:
        abort(404)

    return send_from_directory('.', filename)

@app.route('/')
def index():
    ppicture_path = '.\\images\\unud.png'
    ppicture = readImage(ppicture_path, 140, 140)
    ppicture_path2 = '.\\images\\mipa.png'
    ppicture2 = readImage(ppicture_path2, 140, 140)
    ppicture_path3 = '.\\images\\ilkom.png'
    ppicture3 = readImage(ppicture_path3, 140, 140)
    car1_path = '.\\images\\sekaa gong.jpg'
    car1 = readImage(car1_path, 1024, 400)
    return render_template(
        'home.html',
        pp = ppicture,
        pp2 = ppicture2,
        pp3 = ppicture3,
        car1 = car1
    )

@app.route('/upload', methods=['POST'])
def upload():
    starttime = time.time()
    target = os.path.join(APP_ROOT, 'wavs/')

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist('file'):
        print(file)
        filename = file.filename
        destination = '/'.join([target, filename])
        file.save(destination)
        rates, audio = wavfile.read(destination)
        N = 1024    #FFT LENGTH
        M = int(rates*0.02)         #Analysis window size
        H = int(0.5*M)        #Overlap between windows
        w = get_window("hamming", M)    #Window vector
        audio = np.float32(audio)/norm_fact[audio.dtype.name]   #Samples normalization
        Energy, e_timestamp = energy.root_mean_square(audio, M, rates)  #Define energy per frame
        Esilences = detect_silence(rates, M, Energy)
        backtrack = int(2*(rates/float(M)))
        cuts = np.array([])
        strings = []
        spekt = []
        modspekt = []
        for i in range(Esilences.shape[0]):
            cutpoint = Esilences[i,0]
            if cutpoint - backtrack >= 0:
                cut = audio[(cutpoint - backtrack) * M: cutpoint * M]
                cuts = np.append(cuts, cut)
                mX, pX = stft.stftAnal(cut, rates, w, N, H)
                filename = filename.split('.')[0]
                create_spectrogram(mX, H, rates, N, filename, i)
                spectimg = "models/test/"+filename+"_"+str(i)+".png"
                fakepath = '.\\'+'\\'.join(spectimg.split('/'))
                spekt.append(readImage(fakepath, 160, 120))
                test = cv2.imread(spectimg, 0)
                kpTest, desTest = orb.detectAndCompute(test, None)
                scores = np.array((), dtype = 'int32')
                for m in models:
                    train = cv2.imread(m, 0)
                    kpTest, desTrain = orb.detectAndCompute(train, None)
                    matches = bf.match(desTrain, desTest)
                    matches = sorted(matches, key = lambda x:x.distance)
                    distance = []
                    for j in range(len(matches)):
                        distance.append(matches[j].distance)
                    scores = np.append(scores, np.mean(distance))
                # temp_scores = np.array((), dtype = 'int32')
                # for s in range(5):
                #     temp_scores = np.append(temp_scores, np.min(scores[s*5:(s+1)*5]))
                strings.append(models[np.argmin(scores)].split('/')[-2])
                # print(temp_scores)
                # temp_models = ['a', 'e', 'i', 'o', 'u']
                # strings.append(temp_models[np.argmin(temp_scores)])
                # winner_model = models[np.argmin(temp_scores)*5+np.argmin(scores[np.argmin(temp_scores)*5:np.argmin(temp_scores)*5+5])]
                fakepath2 = '.\\'+'\\'.join(models[np.argmin(scores)].split('/'))
                # fakepath2 = '.\\'+'\\'.join(winner_model.split('/'))
                modspekt.append(readImage(fakepath2, 160, 120))
    result = brute_force(strings)
    endtime = time.time() - starttime
    return render_template('result.html',
                            rates = rates, 
                            result = result, 
                            strings = strings,
                            images = spekt,
                            imglen = len(spekt),
                            modspekt = modspekt,
                            endtime = endtime
                            )

@app.route('/feature')
def feature():
    path = '.\\images\\STEC.png'
    STECimg = readImage(path, 100, 200)
    path2 = '.\\images\\Flowchart Fingerprint.png'
    fingerprint = readImage(path2, 250,100)
    return render_template('feature.html', image1 = STECimg, image2 = fingerprint)

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(port="5897", debug=True)