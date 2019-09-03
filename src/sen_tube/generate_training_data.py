import tarfile
tar = tarfile.open("datasets/SenTube/SenTube.tar")
tar.extractall()
tar.close()