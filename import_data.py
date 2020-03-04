## import_data.py

import gzip;
import numpy as np;
import os;

from struct import unpack;

URL_train_images    =   r'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz';
URL_train_labels    =   r'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz';
URL_test_images     =   r'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz';
URL_test_labels     =   r'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz';

URLs                =   [   URL_train_images,   \
                            URL_train_labels,   \
                            URL_test_images,    \
                            URL_test_labels ];

DATA_PATH           =   os.getcwd() + r'/data/fashion/';
LABEL_MAGIC_NUMBER  =   2049;
IMAGE_MAGIC_NUMBER  =   2051;

def import_urls( URLs = URLs, redownload_data = 0, file_path = DATA_PATH):

    data = []; 

    if redownload_data:

        for url in URLs:

            r   =   requests.get(url);

            if  r.status_code   != 200:
                print(  "File download failed: ", url);
                continue;

            # Retrieve HTTP meta-data
            #print(r.status_code)
            #print(r.headers['content-type'])
            #print(r.encoding)

            file_name   =   file_path + url.split('/')[-1];

            with open(file_name, 'wb') as f:
                f.write(r.content);
    
    
    
    file_names  =   [];

    for (dirpath, dirnames, filenames) in os.walk(file_path):
        for filename in filenames:
            file_names.extend( [file_path + filename]);
    
    #print(file_names)

    ## Method to convert byte-data to int type.
    byte_to_int     =   lambda byt: unpack("<L", byt[::-1]) [0];

    X_train, y_train    =   None, None;
    X_test,  y_test     =   None, None;

    for filename in file_names:

        with gzip.open(filename, 'rb') as f:
            magic_number        =   byte_to_int(f.read(4));
            num_items           =   byte_to_int(f.read(4));
            item_size           =   1;
            data_buf            =   [];

            image_size_row  =   1;
            image_size_col  =   1;
            tag             =   r'label';


            ## Processing Image Data. Magic Number = 0x00000803(2051).
            if magic_number == IMAGE_MAGIC_NUMBER:

                image_size_row  =   byte_to_int(f.read(4));
                image_size_col  =   byte_to_int(f.read(4));

                item_size       =   image_size_row * image_size_col;
                tag             =   r'image';                
            

            buf         = f.read(   num_items * item_size);
            data        = np.frombuffer(    buf, dtype=np.uint8).astype(np.int32);

            if  tag ==  r'image':
                data    = data.reshape(     num_items,  image_size_row, image_size_col);
                if      num_items == 60000:     X_train =   data;
                elif    num_items == 10000:     X_test  =   data;
            
            if  tag ==  r'label':
                data    = data.reshape(     num_items,  image_size_row);
                if      num_items == 60000:     y_train =   data;
                elif    num_items == 10000:     y_test  =   data;

            #label       = np.frombuffer(buf, dtype=np.uint8).astype(np.int64);

    return  (X_train, y_train), (X_test, y_test);