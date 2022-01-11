import datetime
import matplotlib.pyplot
import plotly as pio

format = "%Y-%m-%dT%H_%M_%S"

def validate_data(imgpath, filename):
    now = datetime.datetime.now()
    if imgpath is None:
        raise Exception('Must set image path and a filename')
    elif filename is None:
        filename = 'noname_' + now.strftime(format)
        print('No file name informed, saving with default noname_datetime file name')

    return imgpath, filename


def save_figure(imgpath, filename, image, isfigure=False, is_ax=False):
    now = datetime.datetime.now()
    imgpath, filename = (imgpath, filename)
    filetosave = imgpath + '/' + filename + now.strftime(format)
    for ext in ['png', 'jpeg']:
        if isfigure:
            if is_ax:
                try:
                    image.savefig(imgpath + '/' + filename + now.strftime(format) + f'.{ext}', dpi=300)
                except:
                    pass#print('Cannot save figure yet, must configure orca and electra!')
            else:
                try:
                    image.figure.savefig(imgpath + '/' + filename + now.strftime(format) + f'.{ext}', dpi=300)
                except:
                    pass#print('Cannot save figure yet, must configure orca and electra!')
                    # pio.io.write_image(plt, filetosave, ext)
        else:
            image.savefig(imgpath + '/' + filename + now.strftime(format) + f'.{ext}', dpi=300)
