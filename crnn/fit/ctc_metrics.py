from mxnet.metric import EvalMetric
from mxnet import nd
import numpy as np
class CtcMetrics(EvalMetric):
    def __init__(self, axis=1, name='accuracy',
                 output_names=None, label_names=None,num_classes=11):
        super(CtcMetrics, self).__init__(
            name, axis=axis,
            output_names=output_names, label_names=label_names)
        self.axis = axis
        self.num_classes = num_classes

    def update(self, labels, preds):
        for label, pred_label in zip(labels, preds):
            batch = label.shape[0]
            pred_label = nd.reshape(pred_label,shape =(-4,-1,batch,0))#35,bz,5990
            pred_label = nd.transpose(data=pred_label, axes=[1,0,2])#bz,35,5990
            pred_label = pred_label.asnumpy()
            pred_label = np.argmax(pred_label,axis=2)#bz,35
            pred_label = pred_label.astype('int32')
            label = label.asnumpy().astype('int32')#bz,10
            for i in range(batch):
                ll = label[i]
                ll = ll[np.where(ll!=self.num_classes-1)]#avoid blabk
                pp = pred_label[i]
                pp = pp[np.where(pp!=self.num_classes-1)]#avoid blank
                cut = np.array([])
                #跳过重复
                j = 0
                while j < pp.shape[0]-1:
                    if pp[j+1] == pp[j]:
                        cut = np.append(cut,j+1)
                    j += 1
                pp = np.delete(pp,cut)
                if ll.shape[0] == pp.shape[0]:
                    if np.all(ll == pp):
                        self.sum_metric +=1
            self.num_inst += batch

if __name__ == '__main__':
    pred = nd.array([[0.1,0.2,0.5,0.8,0.3,0.9],
                    [0.9, 0.2, 0.5, 0.8, 0.3, 0.01],
                    [0.1, 0.9, 0.5, 0.8, 0.3, 0.02],
                    [0.1, 0.2, 0.5, 0.001, 0.3, 0.9],
                    [0.1, 0.2, 0.5, 0.8, 0.3, 0.9],
                    [0.1, 0.2, 0.5, 0.8, 0.9, 0.05]])
    print(pred.shape)
    lable = nd.array([[5.0,3.0],
                      [0.0,5.0],
                      [1.0,4.0]])
    print('a')
    ctc = CtcMetrics()
    ctc.update([lable],[pred])
    acc = ctc.get()
    print(acc)