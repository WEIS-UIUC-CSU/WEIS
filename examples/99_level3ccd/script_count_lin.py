import os

ofpath = 'output/OF_lin'
casedir = ['{:08d}'.format(idx) for idx in range(1, 401)]

for idx in range(len(casedir)):
    fullcasedir = os.path.join(ofpath, casedir[idx])
    if not os.path.isdir(fullcasedir):
        print('{:} does not contain .lin files'.format(fullcasedir))
        continue
    listfile = os.listdir(fullcasedir)
    listlinfile = []
    for jdx in range(len(listfile)):
        if len(listfile[jdx]) > 4:
            if listfile[jdx][-4:] == '.lin':
                listlinfile.append(listfile[jdx])
    listlinfile.sort()
    print('{:} contains {:} .lin files'.format(fullcasedir, len(listlinfile)))
