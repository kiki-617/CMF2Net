#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from torch import nn

"""深监督"""
class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None): # weight_factors是权重 [0.53333333 0.26666667 0.13333333 0.06666667 0.  ]
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"  #21个 [(B,3,128,128,128) +  4*(B,3,128,128,128),(B,3,64,64,64),(B,3,32,32,32),(B,3,16,16,16),(B,3,8,8,8)]
        assert isinstance(y, (tuple, list)), "y must be either tuple or list" # 同上
        if self.weight_factors is None: # False
            weights = [1] * len(x)
        else:
            weights = self.weight_factors #26个 [0.53333333]+ 5*[0.53333333 0.26666667 0.13333333 0.06666667 0. ] 实际只用到前21个

        l = weights[0] * self.loss(x[0], y[0]) # tensor(0.8533, device='cuda:0', grad_fn=<MulBackward0>)
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i]) # tensor(1.6297, device='cuda:0', grad_fn=<AddBackward0>)
        return l


