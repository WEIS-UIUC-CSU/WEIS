import os

ofpath = 'output/OF_lin'
casedir = ['{:08d}'.format(idx) for idx in range(1, 401)]

for idx in range(len(casedir)):
    fullcasedir = os.path.join(ofpath, casedir[idx])
    if not os.path.isdir(fullcasedir):
        continue
    listfile = os.listdir(fullcasedir)
    for jdx in range(len(listfile)):
        if len(listfile[jdx]) > 5:
            if listfile[jdx][-5:] == '.outb':
                outfile = os.path.splitext(listfile[jdx])[0] + '.out'
                if os.path.isfile(os.path.join(fullcasedir, outfile)):
                    try:
                        os.remove(os.path.join(fullcasedir, outfile))
                        print('deleting {:} succeeded.'.format(os.path.join(fullcasedir, outfile)))
                    except:
                        print('deleting {:} failed.'.format(os.path.join(fullcasedir, outfile)))

