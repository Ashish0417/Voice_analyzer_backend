import librosa
import numpy as np
import tempfile
from fastapi import UploadFile
import soundfile as sf
import scipy.stats

async def extract_voice_features_from_mp3(file: UploadFile) -> dict:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    # Load audio using librosa
    y, sr = librosa.load(temp_file_path, sr=None)

    # Ensure audio is long enough
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)), mode='constant')

    # Extract features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    sfm = librosa.feature.spectral_flatness(y=y)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    # Fundamental frequency
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[magnitudes > np.median(magnitudes)]
    if len(pitches) == 0:
        pitches = np.array([0])

    # Dominant frequency
    freqs = np.abs(np.fft.rfft(y))
    dom_freqs = np.fft.rfftfreq(len(y), d=1/sr)

    # Features
    meanfreq = np.mean(dom_freqs)
    sd = np.std(dom_freqs)
    median = np.median(dom_freqs)
    Q25 = np.percentile(dom_freqs, 25)
    Q75 = np.percentile(dom_freqs, 75)
    IQR = Q75 - Q25
    skew = scipy.stats.skew(freqs)
    kurt = scipy.stats.kurtosis(freqs)
    sp_ent = scipy.stats.entropy(np.abs(freqs))
    sfm_val = np.mean(sfm)
    mode = dom_freqs[np.argmax(freqs)]
    centroid_val = np.mean(centroid)
    meanfun = np.mean(pitches)
    minfun = np.min(pitches)
    maxfun = np.max(pitches)
    meandom = np.mean(dom_freqs)
    mindom = np.min(dom_freqs)
    maxdom = np.max(dom_freqs)
    dfrange = maxdom - mindom
    modindx = np.std(pitches) / (np.mean(pitches) + 1e-6)

    features = {
        'meanfreq': float(meanfreq),
        'sd': float(sd),
        'median': float(median),
        'Q25': float(Q25),
        'Q75': float(Q75),
        'IQR': float(IQR),
        'skew': float(skew),
        'kurt': float(kurt),
        'sp.ent': float(sp_ent),
        'sfm': float(sfm_val),
        'mode': float(mode),
        'centroid': float(centroid_val),
        'meanfun': float(meanfun),
        'minfun': float(minfun),
        'maxfun': float(maxfun),
        'meandom': float(meandom),
        'mindom': float(mindom),
        'maxdom': float(maxdom),
        'dfrange': float(dfrange),
        'modindx': float(modindx)
    }

    return features
