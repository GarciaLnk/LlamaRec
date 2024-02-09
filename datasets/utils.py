import tarfile
import urllib.request
import zipfile


def download(url, savepath):
    urllib.request.urlretrieve(url, str(savepath))
    print()


def unzip(zippath, savepath):
    print("Extracting data...")
    zip = zipfile.ZipFile(zippath)
    zip.extractall(savepath)
    zip.close()


def unziptargz(zippath, savepath):
    print("Extracting data...")
    f = tarfile.open(zippath)
    f.extractall(savepath)
    f.close()
