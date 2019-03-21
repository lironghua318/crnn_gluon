# decoding: utf-8
import cv2
import mxnet as mx
import numpy as np
from mxnet import gluon,nd

def main():
    ctx = [mx.cpu()]
    model_path = 'model/digit-symbol.json'
    saved_params = 'model/digit-0040.params'
    input_pic = '0159.jpg'
    net = gluon.nn.SymbolBlock.imports(model_path, ['data'], saved_params,ctx = ctx)
    img = cv2.imread(input_pic)
    img = cv2.resize(img,(80,32))
    img = nd.array(img, dtype='float32')
    img = nd.transpose(data=img, axes=[2, 0, 1])
    img = nd.expand_dims(img,axis=0)
    img -= 127.5
    img *= 0.0078125
    print(img.shape)
    pred = net(img)
    batch =img.shape[0]
    pred_label = nd.reshape(pred, shape=(-4, -1, batch, 0))  # 35,bz,5990
    pred_label = nd.transpose(data=pred_label, axes=[1, 0, 2])  # bz,35,5990
    pred_label = pred_label.asnumpy()
    pred_label = np.argmax(pred_label, axis=2)  # bz,35

    char_lists = ['0','1','2','3','4','5','6','7','8','9','blank']
    strs = ""
    for pp in pred_label:
        for i in range(10):
            # print(prob[i])
            max_index =pp[i]
            # print(max_index)
            if i < 10 - 1 and pp[i] == pp[i+1]:
                continue
            if max_index != 10:
                strs += char_lists[max_index]
        print(strs)

if __name__ == '__main__':
    main()




