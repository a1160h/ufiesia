# Functions 順伝播逆伝播双方に対応した関数
# 20260916 A.Inoue

from ufiesia.Config import *
np = Config.np
import copy
from functools import reduce
import itertools

class Assign:
    def forward(self, x):
        return x.copy()  # <要注意>入力と出力は別物にする必要がある

    def backward(self, gy):
        return assign(gy)
    
def assign(x):
    return Assign().forward(x)

class Branch:
    """ 下流へ分岐する際に、下流からの勾配を順に受け取り加算する """
    def __init__(self):
        self.gx = None
        
    def forward(self, x):
        self.x = x
        return x.copy()
    
    def backward(self, gy, *, flush=True):
        x = self.x
        if self.gx is None or flush: 
            self.gx = np.zeros_like(x)
        self.gx += assign(gy)     
        return self.gx
    
def bracch(x):
    return Branch().forward(x)

class Neg:
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy 

def neg(x):
    return Neg().forward(x)

class Pow:
    def __init__(self, c=1):
        self.c = c
        
    def forward(self, x):
        self.x = x
        y = np.power(x, self.c) # 20241019 x**c
        return y

    def backward(self, gy):
        x = self.x
        c = self.c
        gx = gy * c * x ** (c - 1)
        return gx

def pow(x, c):
    return Pow(c).forward(x)

class Square:
    def forward(self, x):
        self.x = x
        y = np.square(x)
        return y

    def backward(self, gy):
        x = self.x
        gx = gy * 2 * x 
        return gx
    
def square(x):
    return Square().forward(x)

class Sqrt:
    def forward(self, x):
        y = np.sqrt(x)
        self.y = y
        return y

    def backward(self, gy):
        y = self.y
        gx = gy * 0.5 / (y + 1e-12) # gy * 0.5 * self.x ** (-0.5) 
        return gx
    
def sqrt(x):
    return SquareRoot()(x)

class Exp:
    """ 指数関数(底を指定可能) """
    def __init__(self, a=None):
        if a is None:          # 底がネイピア数eの場合
            log_of_base = 1          
        else:                  # 底が指定された場合　
            log_of_base = np.log(a) # 底がeの対数,これを用いて底の交換
        self.log_of_base = log_of_base   
   
    def forward(self, x):
        y = np.exp(self.log_of_base * x)
        self.y = y
        return y

    def backward(self, gy):
        y = self.y
        gx = gy * self.log_of_base * y
        return gx

def exp(x, a=None):
    return Exp(a).forward(x)


class Log:
    """ 対数関数(底を指定可能) """
    def __init__(self, a=None):
        if a is None:          # 自然対数
            log_of_base = 1
        elif a > 0 and a != 1: # 対数の底が指定された場合
            log_of_base = np.log(a) 
        else:
            raise Exception("Bad base is given for log")
        self.log_of_base = log_of_base   
       
    def forward(self, x):
        self.x = x
        y = np.log(x)/self.log_of_base
        return y

    def backward(self, gy):
        x = self.x
        gx = gy / (x * self.log_of_base)
        return gx

def log(x, a=None): 
    return Log(a).forward(x) 

class Abs:
    def forward(self, x):
        self.x = x
        y = np.abs(x)
        return y

    def backward(self, gy):
        x = self.x
        gx = gy * ((x >= 0) * 2 - 1)
        return gx

def abs(x):
    return Abs().forward(x)
    
class Sin:
    def forward(self, x):
        self.x = x
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.x
        gx = gy * cos(x)
        return gx

def sin(x):
    return Sin().forward(x)

class Cos:
    def forward(self, x):
        self.x = x
        y = np.cos(x)
        return y

    def backward(self, gy):
        x = self.x
        gx = gy * - sin(x)
        return gx

def cos(x):
    return Cos().forward(x)

class Erf:
    """ 誤差関数(ガウスの誤差関数) """
    def __init__(self):
        if np.__name__ == 'cupy':
            from cupyx.scipy.special import erf
            self.erf = erf
        else:
            try:
                from scipy.special import erf
                self.erf = erf
            except ImportError:
                self.erf = self.AbramowitzStegun

    @staticmethod
    def AbramowitzStegun(x):
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p  = 0.3275911
        sign = np.sign(x)
        ax = np.abs(x)
        t = 1.0 / (1.0 + p * ax)
        poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
        y = 1.0 - poly * np.exp(-ax * ax)
        return sign * y

    def forward(self, x):
        self.x = x
        return self.erf(x)

    def backward(self, gy):
        x = self.x
        return gy * (2.0 / np.sqrt(np.pi)) * np.exp(-x * x)

class Add:
    def forward(self, x0, x1):
        self.x0, self.x1 = x0, x1
        y = x0 + x1
        self.y_shape = y.shape
        return y
    
    def backward(self, gy):
        x0, x1 = self.x0, self.x1
        y_shape = self.y_shape
        x0_shape, x1_shape = np.shape(x0), np.shape(x1)
        gx0 = assign(gy) if y_shape==x0_shape else SumTo(x0_shape).forward(gy)
        gx1 = assign(gy) if y_shape==x1_shape else SumTo(x1_shape).forward(gy)
        return gx0, gx1

def add(x0, x1):
    return Add().forward(x0, x1)

class Sub:
    def forward(self, x0, x1):
        self.x0, self.x1 = x0, x1
        y = x0 - x1
        self.y_shape = y.shape
        return y

    def backward(self, gy):
        x0, x1 = self.x0, self.x1
        y_shape = self.y_shape
        x0_shape, x1_shape = np.shape(x0), np.shape(x1)
        gx0 =  assign(gy) if y_shape==x0_shape else SumTo(x0_shape).forward(gy)
        gx1 = -gy         if y_shape==x1_shape else SumTo(x1_shape).forward(-gy)
        return gx0, gx1

def sub(x0, x1):
    return Sub().forward(x0, x1)

def rsub(x0, x1):
    return Sub().forward(x1, x0)

class Mul:
    def forward(self, x0, x1):
        self.x0, self.x1 = x0, x1
        y = x0 * x1
        self.y_shape = y.shape
        return y
    
    def backward(self, gy):
        x0, x1 = self.x0, self.x1
        y_shape = self.y_shape
        x0_shape, x1_shape = np.shape(x0), np.shape(x1)
        gx0 = x1 * gy
        gx1 = x0 * gy
        gx0 = gx0 if y_shape==x0_shape else SumTo(x0_shape).forward(gx0)
        gx1 = gx1 if y_shape==x1_shape else SumTo(x1_shape).forward(gx1)
        return gx0, gx1
    
def mul(x0, x1):
    return Mul().forward(x0, x1)

class Div:
    def forward(self, x0, x1):
        self.x0, self.x1 = x0, x1
        y = x0 / x1
        self.y_shape = y.shape
        return y
    
    def backward(self, gy):
        x0, x1 = self.x0, self.x1
        y_shape = self.y_shape
        x0_shape, x1_shape = np.shape(x0), np.shape(x1)
        gx0 = gy / x1
        gx1 = - gy * x0 / x1 ** 2
        gx0 = gx0 if y_shape==x0_shape else SumTo(x0_shape).forward(gx0)
        gx1 = gx1 if y_shape==x1_shape else SumTo(x1_shape).forward(gx1)
        return gx0, gx1

def div(x0, x1):
    return Div().forward(x0, x1)

def rdiv(x0, x1):
    return Div().forward(x1, x0)

class SumTo:
    def __init__(self, shape=()):
        self.shape = shape
        self.gy_shape = None

    def forward(self, x):
        self.x = x
        if x.shape == self.shape:
            self.gy_shape = x.shape             # backwardで必要
            return x

        ope_shape = list(self.shape)            # 操作用にコピー
        target_shape = () 
        for i, sx in enumerate(x.shape):        # 次元数を揃えたターゲット形状を得る
            if   len(ope_shape) > i and ope_shape[i] == sx:
                target_shape += sx,
            elif len(ope_shape) > i and ope_shape[i] == 1:
                target_shape += 1,
            else:                               # ope_shape[i] != sxもこのケース
                target_shape += 1,
                ope_shape = [1] + ope_shape     # ope_shapeを1つずらす
        axis = ()
        for i in range(len(x.shape)):           # 同一次元数での比較して軸を得る　
            if target_shape[i] != x.shape[i]:
                axis += i,

        y = np.sum(x, axis=axis, keepdims=True) # 下記の為に次元保持
        self.gy_shape = y.shape                 # backwardで必要(操作したxの形状に)
        #y = y.reshape(self.shape)
        y = np.reshape(y, self.shape)
        return y

    def backward(self, gy):
        x = self.x
        #gx = gy.reshape(self.gy_shape)          # 先ずは次元数を合わせる
        gx = np.reshape(gy, self.gy_shape)      # 先ずは次元数を合わせる
        gx = np.broadcast_to(gx, x.shape)       # それから所望のbroadcast
        return gx

def sum_to(x, shape):
    return SumTo(shape).forward(x)

class BroadcastTo:
    def __init__(self, shape=()):
        self.shape = shape

    def forward(self, x):
        self.x = x
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        x = self.x
        gx = sum_to(gy, np.shape(x))
        return gx

def broadcast_to(x, shape):
    return BroadcastTo(shape).forward(x)

class SumMeanVar:
    """ 配列操作を担う共通クラス """
    def __init__(self, axis=None, dtype=None, out=None, keepdims=False):
        self.keepdims = keepdims
        self.axis = axis
        self.n = None
        self.gy_shape = None
        self.dtype = dtype # 仮処置20260127AI
        self.out = out     # 仮処置20260127AI
        self.last_x_shape = None
    
    def set_axis_and_shape(self, shape):
        """ 畳まれる軸と残る軸を明らかにする """
        ndim = len(shape)
        axis = self.axis # 指定された軸
        all_axis = list(range(ndim))
        if axis is None:
            axis = all_axis
        if type(axis) not in (tuple, list):
            #print('axisをタプルに')
            axis = axis,
            
        # 畳まれる軸と畳まれない軸    
        axis = [ndim + ax if ax<0 else ax for ax in axis] # 負値に対応

        # 畳まれる軸を1、他はそのままの形状
        self.gy_shape = [1 if ax in axis else s for ax, s in zip(all_axis, shape)]

        n = 1
        for ax in axis:
            n *= shape[ax] # 畳まれる軸の形状の積=大きさ
        #print('指定された軸', self.axis)
        #print('畳まれる軸', axis)
        #print('畳まれる軸を1、他はそのままの形状', self.gy_shape)
        #print('畳まれる軸の形状の積=大きさ', n)
        self.n = n
        
    def backward(self, gy):
        x = self.x
        x_shape = np.shape(x)
        if self.last_x_shape != x_shape:
            self.set_axis_and_shape(x_shape)
            self.last_x_shape = x_shape
        gy = np.reshape(gy, self.gy_shape)         # gyは次元を合わせる
        gy = np.broadcast_to(gy, x_shape)          # 畳まれた分をbroadcastして元に戻す
        return gy


class Sum(SumMeanVar):
    """ 和 """
    def forward(self, x):
        self.x = x
        y = np.sum(x, axis=self.axis, keepdims=self.keepdims)
        return y

def sum(x, axis=None, keepdims=False):
    return Sum(axis=axis, keepdims=keepdims).forward(x)

class Mean(SumMeanVar):
    """ 平均 """
    def forward(self, x):
        self.x = x
        y = np.mean(x, axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        gy = super().backward(gy)
        gx = gy * (1/self.n) 
        return gx

def mean(x, axis=None, dtype=None, out=None, keepdims=False):
    return Mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims).forward(x)

class Var(SumMeanVar):
    """ 分散 """
    def forward(self, x):
        self.x = x
        y = np.var(x, axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = super().backward(gy)
        x = self.x
        mu = np.mean(x, axis=self.axis, keepdims=True)
        gx = gy * (2/self.n) * (x - mu)
        return gx

def var(x, axis=None, keepdims=False):
    return Var(axis=axis, keepdims=keepdims).forward(x)

class Std(SumMeanVar):
    """ 標準偏差 """
    def forward(self, x):
        self.x = x
        y = np.std(x, axis=self.axis, keepdims=self.keepdims)
        self.y = y
        return y

    def backward(self, gy):
        gy = super().backward(gy)
        x = self.x
        y = self.y
        mu = np.mean(x, axis=self.axis, keepdims=True)
        yr = np.reshape(y, self.gy_shape)
        gvar = gy * 0.5 / (yr + 1e-12) # std = sqrt(var) の逆伝播　
        dvar_dx = (2/self.n) * (x - mu)
        gx = gvar * dvar_dx
        return gx

def std(x, axis=None, keepdims=False):
    return Std(axis=axis, keepdims=keepdims).forward(x)

class SquareSum(SumMeanVar):
    """ 二乗和 """
    def forward(self, x):
        self.x = x
        y = np.sum(x**2, axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = super().backward(gy)
        x = self.x
        gx = gy * 2 * x
        return gx
        
class SquareMean(SumMeanVar):
    """ 二乗平均 """
    def forward(self, x):
        self.x = x
        y = np.mean(x**2, axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = super().backward(gy)
        x = self.x
        gx = gy * (1/self.n) * 2 * x
        return gx
    
class RootSumSquare(SumMeanVar):
    """ 二乗和平方根 """
    def forward(self, x):
        self.x = x
        sqsm = np.sum(x**2, axis=self.axis, keepdims=self.keepdims)
        y = np.sqrt(sqsm)
        self.y = y
        return y

    def backward(self, gy):
        gy = super().backward(gy)
        x = self.x
        y = self.y
        yr = np.reshape(y, self.gy_shape) # gyと形状を揃える
        gsqsm = gy * 0.5 / (yr + 1e-12)    # RMS = sqrt(sqmu)の逆伝播
        gx = 2 * x * gsqsm
        return gx

class RootMeanSquare(SumMeanVar):
    """ 二乗平均平方根(RootMeanSquare) """
    def forward(self, x):
        self.x = x
        sqmu = np.mean(x**2, axis=self.axis, keepdims=self.keepdims)
        y = np.sqrt(sqmu)
        self.y = y
        return y

    def backward(self, gy):
        gy = super().backward(gy)
        x = self.x
        y = self.y
        yr = np.reshape(y, self.gy_shape) # gyと形状を揃える
        gsqmu = gy * 0.5 / (yr + 1e-12)    # RMS = sqrt(sqmu)の逆伝播
        gx = (1/self.n) * 2 * x * gsqmu
        return gx

class RMS(RootMeanSquare):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class VariadicBase:
    def __init__(self):
        self.func = None
        self.packed_in_one = None

    def forward(self, *xs): # 引数の数は不定
        if all(isinstance(x, np.ndarray) for x in xs):    
            y = self.func.forward(*xs)
            self.packed_in_one = False
            
        elif len(xs)==1 and all(isinstance(x, (tuple, list)) for x in xs):
            xs, = xs
            y = self.func.forward(*xs)
            self.pcked_in_one =True
            
        elif len(xs)==1 and all(isinstance(x, type((i for i in []))) for x in xs):
            y = self.func.forward(tuple(xs[0]))
            self.packed_in_one = True
        else:
            raise Exception('Non-compliant input data.')
        return y

    def __call__(self, *xs):
        return self.forward(*xs)

    def backward(self, gy=1):
        gxs = self.func.backward(gy)
        return gxs[0] if len(gxs)==1 else gxs
    
class SumVariadicCore:
    def forward(self, *xs):
        #print('->', self.__class__.__name__, type(xs), len(xs), '\n', xs)
        y = reduce(lambda a, b : a + b, xs)
        self.reduction = len(xs)
        return y

    def backward(self, gy):
        gxs = (gy,) * self.reduction
        return gxs

class SumVariadic(VariadicBase):
    def __init__(self):
        super().__init__()
        self.func = SumVariadicCore()

def sum_variadic(*xs):
    return SumVariadic().forward(*xs)


class MaxMin:
    """ MaxとMinの共通ベース """
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def condition(self, x, y, axis):
        """ 抽出されたところを示す配列を作る、併せて逆伝播に必要なものを揃える """
        self.x_shape = x.shape
        self.y_shape = y.shape
        if axis is None:
            axis = range(x.ndim)
        elif isinstance(axis, int):
            axis = axis,                      
        self.z_shape = [1 if i in axis else s for i, s in enumerate(x.shape)] #xと同次元数のyの形
        #self.cond = x == y.reshape(self.z_shape) # xとyの比較
        self.cond = x == np.reshape(y, self.z_shape) # xとyの比較

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def backward(self, gy):
        """ 逆伝播は共通 """
        gy = gy if isinstance(gy, np.ndarray) else np.array(gy, dtype=Config.dtype) 
        gy = np.broadcast_to(gy, self.y_shape)  # 先ずはyの形状に合わせる
        gy = np.reshape(gy, self.z_shape)       # 次にxに次元を揃える(keepdimsの形状)
        gx = gy * self.cond                      # yに抽出されたところにgyを入れる
        return gx

class Max(MaxMin):
    """ 最大値を抽出 """
    def forward(self, x):
        y = np.max(x, axis=self.axis, keepdims=self.keepdims)
        self.condition(x, y, self.axis)
        return y

class Min(MaxMin):
    """ 最小値を抽出 """
    def forward(self, x):
        y = np.min(x, axis=self.axis, keepdims=self.keepdims)
        self.condition(x, y, self.axis)
        return y

     
class GetItem:
    """ 要素を添字指定により部分取り出しする """
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        self.x = x
        return x[self.slices]

    def backward(self, gy):
        x = self.x
        gx = np.zeros_like(x, dtype=Config.dtype)
        np.add.at(gx, self.slices, gy)
        return gx

def getitem(x, slices):
    f = GetItem(slices)
    return f.forward(x)


class TopKprimitive:
    """
    指定軸axisに沿って、大きい順に上位k個の値を返す


    TopKの順伝播は、逆伝播の対象外であるindicesも返すため、
    primitiveではindicesを返り値から除外し、valuesのみを返す

    """
    def __init__(self, k, axis=-1):
        if k < 1:
            raise ValueError(f"k must be at least 1, but got {k}")
        self.k = k
        self.axis = axis
        self.indices = None

    def forward(self, x):
        self.x = x
        axis = self.axis

        # 軸axisに沿って降順でxのインデクスを並べ先頭のk個を選ぶ
        indices = np.argsort(x, axis=axis)
        indices = np.flip(indices, axis=axis)
        indices = np.take(indices, np.arange(self.k), axis=axis)

        # indicesとaxisに従って値を選ぶ
        values = np.take_along_axis(x, indices, axis=axis)
        self.indices = indices
        return values

    def backward(self, gy):
        x = self.x
        gx = scatter_add_along_axis(
            gy, self.indices, output_shape=x.shape, axis=self.axis,
        )
        return gx

class TopK:
    """
    指定軸axisに沿って、大きい順に上位k個の値とindicesを返す

    逆伝播の対象となるvaluesのみを返すprimitiveをラッパーで包み、
    逆伝播の対象外であるindicesを加えて提供する

    """
    def __init__(self, k, axis=-1):
        super().__init__()
        self.primitive = TopKprimitive(k, axis)

    def forward(self, x):
        y = self.primitive.forward(x)
        indices = self.primitive.indices # indicesは逆伝播非対象
        return y, indices

    def __call__(self, x):
        return self.forward(x)

    def backward(self, gy=1):
        return self.primitive.backward(gy)

def top_k(x, k, axis=-1):
    return TopK(k, axis).forward(x)


class Transpose_bkup:
    def __init__(self, axes=(1, 0)): # numpyのおかしな挙動に対応20241122
        if len(axes)==1:
            self.axes, = axes
        else:    
            self.axes = axes

    def forward(self, x):
        y = np.transpose(x, self.axes)
        return y

    def backward(self, gy):
        axes = np.argsort(np.array(self.axes)) # cupy対応
        gx = np.transpose(gy, axes.tolist())   # cupy対応 
        return gx

class Transpose:
    def __init__(self, *axes): # axesがタプルでなくても対応
        #print('###', axes)
        if axes is None:
            self.axes = (1, 0)
        elif len(axes)==0:
            self.axes = (1, 0)
        elif len(axes) > 1:    
            self.axes = axes
        else: # タプルの中にタプル
            self.axes, = axes

    def forward(self, x):
        y = np.transpose(x, self.axes)
        return y

    def backward(self, gy):
        axes = np.argsort(np.array(self.axes)) # cupy対応
        gx = np.transpose(gy, axes.tolist())   # cupy対応 
        return gx

def transpose(x, *axes):
    return Transpose(*axes).forward(x)

def transpose_bkup(x, axes=(1, 0)):
    return Transpose(axes).forward(x)

class Reshape_bkup:
    def __init__(self, *shape):
        if len(shape) > 1:
            self.shape = shape
        else:
            self.shape, = shape # タプルにする
        
    def forward(self, x):
        self.x = x
        y = np.reshape(x, self.shape)
        return y

    def backward(self, gy):
        x = self.x
        gx = np.reshape(gy, np.shape(x))
        return gx

class Reshape:
    def __init__(self, *shape):

        # 正規化：常に tuple[int,...] にする
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            self.shape = tuple(shape[0])
        else:
            self.shape = shape

    def forward(self, x):
        self.x = x
        return np.reshape(x, self.shape)

    def backward(self, gy):
        x = self.x
        return np.reshape(gy, np.shape(x))

def reshape(x, *shape):
    return Reshape(*shape).forward(x)

class Dot:
    def forward(self, x0, x1):
        self.x0, self.x1 = x0, x1
        y = np.dot(x0, x1)
        return y

    def backward(self, gy):
        x0, x1 = self.x0, self.x1
        gx0 = np.dot(gy, x1.T)
        gx1 = np.dot(x0.T, gy)
        return gx0, gx1
     
def dot(x0, x1):
    return Dot().forward(x0, x1)

class MatMul:
    def forward(self, x0, x1):
        self.x0, self.x1 = x0, x1
        y = np.matmul(x0, x1)
        return y

    def backward(self, gy):
        x0, x1 = self.x0, self.x1
        #x0T = x0.T if x0.ndim <= 2 else x0.transpose(*range(x0.ndim)[:-2], -1, -2)
        #x1T = x1.T if x1.ndim <= 2 else x1.transpose(*range(x1.ndim)[:-2], -1, -2)
        x0T = x0.T if x0.ndim <= 2 else np.transpose(x0, (*range(x0.ndim)[:-2], -1, -2))
        x1T = x1.T if x1.ndim <= 2 else np.transpose(x1, (*range(x1.ndim)[:-2], -1, -2))
        gx0 = np.matmul(gy, x1T)
        gx1 = np.matmul(x0T, gy)
        return gx0, gx1
     
def matmul(x0, x1):
    return MatMul().forward(x0, x1)

class DotLinear:
    """ ニューラルネットワークで使う基本の重み付け和 """
    def __init__(self, bias=True):
        self.bias = bias
        
    def forward(self, x, w, b):
        self.x, self.w, self.b = x, w, b
        y = np.dot(x, w)
        if self.bias:
            y += b
        return y

    def backward(self, gy):
        x, w, b = self.x, self.w, self.b
        gx = dot(gy, w.T)
        gw = dot(x.T, gy)
        if self.bias:
            gb = np.sum(gy, axis=0)
        else:
            gb = None
        return gx, gw, gb

    def backwardbkup(self, gy):
        # ニューラルネットワークで使うことに限れば不要
        x, w, b = self.x, self.w, self.b
        w_T = transpose(w) 
        x_T = transpose(x)  
        gx = np.dot(gy, w_T)
        gw = np.dot(x_T, gy)
        if self.bias and gy.shape==b.shape:
            gb = gy
        elif self.bias:
            gb = sum_to(gy, b.shape)
        else:
            gb = None
        return gx, gw, gb

class HadamardLinear:
    def forward(self, x, w, b):
        self.x, self.w, self.b = x, w, b
        y = x * w + b
        return y

    def backward(self, gy):
        x, w, b = self.x, self.w, self.b
        gx = gy * w
        gw = gy * x
        gb = gy
        return gx, gw, gb

    def backwardbkup(self, gy):
        # ニューラルネットワークで使うことに限れば不要
        x, w, b = self.x, self.w, self.b
        gx = gy * w
        gw = gy * x
        gb = gy if gy.shape==b.shape else SumTo(b.shape).forward(gy)
        return gx, gw, gb
     
class MatMulLinear:
    """ ニューラルネットワークで使う時系列データなど入力次元数3の重み付け和 """
    def __init__(self, bias=True):
        self.bias = bias
        
    def forward(self, x, w, b):
        self.x, self.w, self.b = x, w, b
        y = np.matmul(x, w)
        if self.bias:
            y += b
        return y

    def backward(self, gy):
        x, w, b = self.x, self.w, self.b
        #x_T = x.T if x.ndim <= 2 else x.reshape(-1, x.shape[-1]).T
        x_T = x.T if x.ndim <= 2 else np.reshape(x, (-1, x.shape[-1])).T
        gx = np.matmul(gy, w.T)
        #gyf = gy.reshape(-1, gy.shape[-1])
        gyf = np.reshape(gy, (-1, gy.shape[-1]))
        gw = np.dot(x_T, gyf)
        if self.bias:
            gb = np.sum(gyf, axis=0)
        else:
            gb = None
        return gx, gw, gb

class MatMulLinear_bkup:
    """ ニューラルネットワークで使う時系列データなど入力次元数大の重み付け和 """
    def __init__(self, bias=True):
        self.bias = bias
        
    def forward(self, x, w, b):
        self.x, self.w, self.b = x, w, b
        y = np.matmul(x, w)
        if self.bias:
            y += b
        return y

    def backward(self, gy):
        # wやbの次元数
        x, w, b = self.x, self.w, self.b
        x_T = x.T if x.ndim <= 2 else np.transpose(x, (*range(x.ndim)[:-2], -1, -2))
        gx = np.matmul(gy, w.T)
        gw = np.matmul(x_T, gy)
        if self.bias:
            gb = np.sum(gy, axis=0)
        else:
            gb = None
        return gx, gw, gb

class DualDotLinear:
    """ ニューラルネットワークで使う２重の重み付け和 """
    def __init__(self, bias=True):
        self.bias = bias
        
    def forward(self, x, r, w, v, b):
        self.x, self.r, self.w, self.v, self.b = x, r, w, v, b
        y = np.dot(x, w) + np.dot(r, v) 
        if self.bias:
            y += b
        return y

    def backward(self, gy):
        x, r, w, v, b = self.x, self.r, self.w, self.v, self.b
        gx = np.dot(gy, w.T)
        gr = np.dot(gy, v.T)
        gw = np.dot(x.T, gy)
        gv = np.dot(r.T, gy)
        if self.bias:
            gb = np.sum(gy, axis=0)
        else:
            gb = None
        return gx, gr, gw, gv, gb

class ScaleDotLinear:
    """ ニューラルネットワークで使う基本の重み付け和、ReParameterization対応 """
    def __init__(self, matmul=False, bias=True, scale=False, eps=1e-8):
        self.matmul = matmul 
        self.bias = bias
        self.scale = scale
        self.dot = np.matmul if matmul else np.dot
        self.eps = eps
        
    def forward(self, x, w, b, g=1.0):
        self.x, self.w, self.b, self.g = x, w, b, g
        y = self.dot(x, w)
        if self.scale:
            y *= g 
        if self.bias:
            y += b
        self.y = y
        return y

    def backward(self, gy):
        x, w, b, g = self.x, self.w, self.b, self.g
        y = self.y
        #x_T = x.T if x.ndim <= 2 else x.reshape(-1, x.shape[-1]).T
        x_T = x.T if x.ndim <= 2 else np.reshape(x, (-1, x.shape[-1])).T
        #gyf = gy.reshape(-1, gy.shape[-1])
        gyf = np.reshape(gy, (-1, gy.shape[-1]))
        
        gx = self.dot(gy, w.T)
        gw = self.dot(x_T, gyf)
        if self.scale:
            gx *= g
            gw *= g
        
        if self.bias:
            gb = np.sum(gyf, axis=0)
        else:
            gb = None

        if self.scale:
            if np.abs(g) > self.eps: # 通常
                z = y - b if self.bias else y # z=g*(x@w)
                gg = np.sum(z * gy) / g       # gg=Σgy*(x@w)
            else:                    # gが0に近いときだけ再計算
                xw = self.dot(x, w)                    
                gg = np.sum(gy * xw)
        else:
            gg = None
        return gx, gw, gb, gg

class Flatten:
    """ 軸0はバッチとし、それ以下の軸を平坦化 """
    def forward(self, x):
        self.x_shape = x.shape
        #return x.reshape(self.x_shape[0], -1)
        return np.reshape(x, (self.x_shape[0], -1))

    def backward(self, gy):
        #return gy.reshape(*self.x_shape)
        return np.reshape(gy, self.x_shape)
        
class Normalize:
    """ 平均0標準偏差1にする標準化(正規化の一種) """
    def __init__(self, axis=None, eps=1e-12):
        self.axis = axis
        self.eps = eps
        self.sigma = None
        self.mask = None
    
    def forward(self, x):
        self.x = x
        mu = np.mean(x, axis=self.axis, keepdims=True)
        sigma = np.std(x, axis=self.axis, keepdims=True)
        z = x - mu
        y = z / (sigma + self.eps)
        self.sigma = sigma
        self.mask = sigma < self.eps # sigmaが極小値の場合には正規化しない
        self.n = x.size//sigma.size  # 畳んだ大きさ 
        y = y * (1 - self.mask) + x * self.mask
        self.y = y
        return y
   
    def backward(self, gy):
        x = self.x
        y = self.y
        sigma = self.sigma + self.eps
        n = self.n
        # y = z / sigma の逆伝播 
        gz = gy / sigma
        gsigma = -gy * y / sigma   # -gy * z / sigma ** 2
        #gsigma = np.sum(gsigma, axis=self.axis, keepdims=True)
        gsigma = sum(gsigma, axis=self.axis, keepdims=True)
        # z = x - mu の逆伝播
        gx0 = gz
        #gmu = np.sum(-gz, axis=self.axis, keepdims=True)
        gmu = sum(-gz, axis=self.axis, keepdims=True)
        # mu = np.mean(x) の逆伝播
        gx1 = np.broadcast_to(gmu, x.shape) / n
        #gx1 = broadcast_to(gmu, x.shape) / n
        # sigma = np.std(x) の逆伝播
        gx2 = gsigma * y / n   # dvar * dvar_gx = (gsigma * 0.5 / sigma) * ((2/n) * (x - mu))
        gx = gx0 + gx1 + gx2
        return gx * (1 - self.mask) + gy * self.mask

class NormalizeSimple:
    """ 平均0標準偏差1にする標準化(正規化の一種) """
    def __init__(self, axis=None, eps=1e-12):
        self.axis = axis
        self.eps = eps
    
    def forward(self, x):
        self.x = x
        mu = np.mean(x, axis=self.axis, keepdims=True)
        sigma = np.std(x, axis=self.axis, keepdims=True)
        z = x - mu
        y = z / (sigma + self.eps)
        self.sigma = sigma
        self.y = y
        return y
   
    def backward(self, gy):
        x = self.x
        y = self.y
        sigma = self.sigma + self.eps
        n = x.size//sigma.size   # 畳んだ大きさ
        # y = z / sigma の逆伝播 
        gz = gy / sigma
        gsigma = -gy * y / sigma   # -gy * z / sigma ** 2
        gsigma = np.sum(gsigma, axis=self.axis, keepdims=True)
        # z = x - mu の逆伝播
        gx0 = gz
        gmu = np.sum(-gz, axis=self.axis, keepdims=True)
        # mu = np.mean(x) の逆伝播
        gx1 = np.broadcast_to(gmu, x.shape) / n
        # sigma = np.std(x) の逆伝播
        gx2 = gsigma * y / n   # dvar * dvar_gx = (gsigma * 0.5 / sigma) * ((2/n) * (x - mu))
        return gx0 + gx1 + gx2

class Normalize_bkup:
    """ 平均0標準偏差1にする標準化(正規化の一種) """
    def __init__(self, axis=None):
        self.mean = Mean(axis, keepdims=True)
        self.std  = Std(axis, keepdims=True)
        self.sub  = Sub()
        self.div  = Div()
    
    def forward(self, x):
        mu  = self.mean.forward(x)
        std = self.std.forward(x)
        z = self.sub.forward(x, mu)
        y = self.div.forward(z, std)
        return y
   
    def backward(self, gy):
        gz, gstd = self.div.backward(gy)
        gx0, gmu = self.sub.backward(gz)
        gx1 = self.mean.backward(gmu)
        gx2 = self.std.backward(gstd)
        gx = gx0 + gx1 + gx2
        return gx

class Standardize(Normalize):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class L2Normalize:
    """ L2ノーマライゼーション """
    def __init__(self, axis=None):
        self.root_sum_square = RootSumSquare(axis=axis, keepdims=True)
        self.div = Div()
    
    def forward(self, x):
        l2n = self.root_sum_square.forward(x)
        y = self.div.forward(x, l2n)
        return y
   
    def backward(self, gy):
        gx0, gl2n = self.div.backward(gy)
        gx1 = self.root_sum_square.backward(gl2n)
        gx = gx0 + gx1
        return gx

class Normalize_bkup:
    """ 平均0標準偏差1にする標準化(正規化の一種) """
    def __init__(self, axis=None):
        self.axis = axis
        
    def forward(self, x):
        self.x = x
        mu =  np.mean(x, axis=self.axis, keepdims=True)
        std = np.std(x, axis=self.axis, keepdims=True)
        self.mu   = mu
        self.std  = std
        y = (x - mu) / (std + 1e-12)
        self.y = y
        return y
    
    def backward(self, gy=1):
        istd = 1/self.std
        iN = self.mu.size / self.x.size # muおよびstdを求める際に畳んだ大きさ
        xc = self.x - self.mu
        gy_sum = np.sum(gy * xc, axis=self.axis, keepdims=True)
        gz = (gy - (self.y * gy_sum * istd * iN)) * istd
        gz_sum = np.sum(gz, axis=self.axis, keepdims=True)
        gx = gz - (gz_sum * iN)
        return gx

class L2Normalize_bkup:
    """ L2ノーマライゼーション """
    def __init__(self, axis=None):
        self.axis = axis
    
    def forward(self, x):
        x = np.array(x)
        l2n = np.sum(x**2, axis=self.axis, keepdims=True)**0.5
        y = x / l2n
        self.x = x
        self.l2n = l2n
        return y
   
    def backward(self, gy=1):
        x = self.x
        l2n = self.l2n
        gx = gy * (1 - x * x.sum(axis=self.axis, keepdims=True) / l2n**2) / l2n
        return gx


class Concatenate:
    """ 複数の入力を、既存の指定軸に沿って結合する """
    def __init__(self, axis=0):
        self.axis = axis

    def forward(self, *xs):
        self.xs = xs
        y = np.concatenate(xs, axis=self.axis)
        self.y_shape = y.shape
        return y

    def backward(self, gy):
        axis = self.axis
        if axis < 0:
            axis += len(self.y_shape)

        sections = []
        stop = 0

        for x in self.xs[:-1]:
            stop += x.shape[axis]
            sections.append(stop)

        return tuple(np.split(gy, sections, axis=axis))

def concatenate(xs, axis=0):
    return Concatenate(axis).forward(*xs)

class ECat(Concatenate):
    def __init__(self, axis=0):
        super().__init__(axis=axis)
        print('Retain for the time being for compatibility. Use Concatenate.')
    
def ecat(*xs, axis=0):
    return ECat(axis).forward(*xs)

class Split:
    """入力を、指定軸に沿って複数に分割する"""
    def __init__(self, indices_or_sections, axis=0):
        self.indices_or_sections = indices_or_sections
        self.axis = axis

    def forward(self, x):
        return np.split(x, self.indices_or_sections, axis=self.axis)

    def backward(self, *gys):
        return np.concatenate(gys, axis=self.axis)

def split(x, indices_or_sections, axis=0):
    return Split(indices_or_sections, axis).forward(x)


class Stack:
    """ 複数の同形状入力を、新しい指定軸に沿って積み重ねる """
    def __init__(self, axis=0):
        self.axis = axis

    def forward(self, *xs):
        return np.stack(xs, axis=self.axis)

    def backward(self, gy):
        return tuple(np.moveaxis(gy, self.axis, 0))

def stack(xs, axis=0):
    return Stack(axis).forward(*xs)


class Unstack:
    """ 入力を指定軸に沿って分解し、その軸を除いた複数の出力を返す """
    def __init__(self, axis=0):
        self.axis = axis

    def forward(self, x):
        return tuple(np.moveaxis(x, self.axis, 0))

    def backward(self, *gys):
        return np.stack(gys, axis=self.axis)

def unstack(x, axis=0):
    return Unstack(axis).forward(x)

def _along_axis_index(indices, axis, input_shape, output_shape):
    """
    AlongAxis操作のScatterAdd用index tupleを作る。

    indicesは出力形状へbroadcastする。
    input側でbroadcastされた軸の座標は0とする。
    """
    indices = np.broadcast_to(indices, output_shape)
    ndim = len(input_shape)
    index = []

    for i, size in enumerate(output_shape):
        if i == axis:
            index.append(indices)

        elif input_shape[i] == 1:
            index.append(0)

        else:
            shape = [1] * ndim
            shape[i] = size

            index.append(np.arange(size).reshape(shape))

    return tuple(index)


class TakeAlongAxis:
    """
    指定軸に沿ってindicesで指定された要素を取り出す
    
    順伝播はnp.take_along_axisそのもの．indicesとaxisは微分対象外
    要注意：axis=Noneの場合、xを一次元化して処理
    """
    def __init__(self, indices, axis=-1):
        self.indices = np.asarray(indices)
        self.axis = axis

    def forward(self, x):
        self.x = x
        y = np.take_along_axis(x, self.indices, axis=self.axis)
        if self.axis is not None and self.axis < 0:
            self.axis += x.ndim
        return y

    def backward(self, gy):
        x = self.x
        gx = np.zeros_like(x, dtype=Config.dtype)

        if self.axis is None: # axis=Noneではxを一次元に展開して処理
            gx_flat = np.reshape(gx, -1)
            np.add.at(gx_flat, self.indices, gy)
        else:
            index = _along_axis_index(self.indices, self.axis, x.shape, gy.shape)
            np.add.at(gx, index, gy)
        return gx

def take_along_axis(x, indices, axis=-1):
    return TakeAlongAxis(indices, axis).forward(x)

class ScatterAddAlongAxis:
    """
    xをindicesで指定されたaxis上の位置へ加算配置する

    重複するindexには値を累積、TakeAlongAxisの随伴演算
    要注意：整数axisのみを扱う
    """
    def __init__(self, indices, output_shape, axis=-1):
        self.indices = np.asarray(indices)
        self.output_shape = tuple(output_shape)
        self.axis = axis

    def forward(self, x):
        self.x = x
        y = np.zeros(self.output_shape, dtype=Config.dtype)

        if x.ndim != y.ndim:
            raise ValueError(
                "x and output must have the same number of dimensions"
            )

        axis = self.axis
        if axis < 0:
            axis += x.ndim
        if axis < 0 or axis >= x.ndim:
            raise ValueError(
                f"axis {self.axis} is out of bounds for array "
                f"of dimension {x.ndim}"
            )

        for i, size in enumerate(x.shape):
            if i != axis and y.shape[i] not in (1, size):
                raise ValueError(
                    "x and output shapes must match outside axis, "
                    "except where the output size is 1"
                )

        index = _along_axis_index(self.indices, axis, y.shape, x.shape)
        np.add.at(y, index, x)

        self.axis = axis
        return y

    def backward(self, gy):
        x = self.x
        indices = np.broadcast_to(self.indices, x.shape)
        gx = np.take_along_axis(gy, indices, axis=self.axis)
        return gx


def scatter_add_along_axis(x, indices, output_shape, axis=-1):
    return ScatterAddAlongAxis(indices, output_shape, axis).forward(x)



############################################ 
# 仮実装　0240812
# MultiHeadAttentionでbacktraceできるようにするために
# その仮対処
############################################
class Tile:
    def __init__(self, reps):
        self.reps = reps  # 繰り返し数を指定するタプルまたは整数。
        self.x_ndim = None
        self.x_shape = None

    def forward(self, x):
        y = np.tile(x, self.reps)
        self.x_ndim = x.ndim
        self.x_shape = x.shape
        return y

    def backward(self, gy):
        #gx = gy.reshape(self.x_shape + self.reps) # 元の形に戻すため
        gx = np.reshape(gy, (self.x_shape + self.reps)) # 元の形に戻すため
        #gx = f.sum_to(gy, self.x_shape)          # 以下は、この動作と同じ
        for ax in self.reps:                      # repsの軸ごとに
            gx = np.sum(gx, axis=self.x_ndim) 
        return gx

def tile(x, reps):
    return Tile(reps).forward(x)

class Pairwise:
    """ 指定された軸に従ってペアを作る（自己ペアのマスクも可能）"""
    def __init__(self, axis=-1, broadcast=True, diagonal_mask=False):
        self.axis = axis
        self.broadcast = broadcast
        self.mask = diagonal_mask
        self.ne = None

    def forward(self, x):
        axis = self.axis % x.ndim
        p = np.expand_dims(x, axis)
        q = np.expand_dims(x, axis + 1)

        if self.broadcast:
            p, q = np.broadcast_arrays(p, q)

        if self.mask:
            sz = x.shape[axis]
            eye = np.eye(sz, dtype=bool)
            mask = ~eye
            shape = [1] * p.ndim
            shape[axis] = sz
            shape[axis + 1] = sz
            self.ne = np.reshape(mask, shape)

            p = np.where(self.ne, p, 0)
            q = np.where(self.ne, q, 0)

        return p, q

    def backward(self, gp, gq):
        if self.mask and self.ne is not None:
            gp = np.where(self.ne, gp, 0)
            gq = np.where(self.ne, gq, 0)

        ndim = gp.ndim
        axis = self.axis % (ndim - 1)

        gxp = np.sum(gp, axis=axis)
        gxq = np.sum(gq, axis=axis + 1)
        gx = gxp + gxq
        return gx

class Pairwise_bkup:
    """ 指定された軸に従ってペアを作る """
    def __init__(self, axis=-1, broadcast=True, diagonal_mask=True):
        self.axis = axis # 後ろ(-1)から数えてペアを作る軸を指定
        self.broadcast = broadcast
        self.mask = diagonal_mask

    def forward(self, x):
        p = np.expand_dims(x, self.axis)
        q = np.expand_dims(x, self.axis-1)
        if self.broadcast:
            p, q = np.broadcast_arrays(p, q)
        if self.mask:
            self.ne = ~np.eye(x.shape[self.axis], dtype=bool)
            p *= self.ne
            q *= self.ne
        return p, q

    def backward(self, gp, gq):
        if self.mask:
            gp *= self.ne
            gq *= self.ne
        gxp = np.sum(gp, axis=self.axis)
        gxq = np.sum(gq, axis=self.axis-1)
        gx = gxp + gxq
        return gx

class UpperTriangle:
    def __init__(self, k=0):
        """ 末尾2軸の正方行列の上三角行列 """
        self.k = k       # 対角線からのオフセット
        self.mask = None # 

    def forward(self, x):
        self.x = x
        if x.shape[-1] != x.shape[-2]:
            raise ValueError("末尾2軸が正方行列である必要があります")
        N = x.shape[-1]
        triu_rows, triu_cols = np.triu_indices(N, k=self.k)
        self.mask = np.zeros((N, N), dtype=bool)
        self.mask[triu_rows, triu_cols] = True 
        y = x[..., self.mask]
        return y

    def backward(self, gy):
        x = self.x
        N = x.shape[-1]
        gx = np.zeros(x.shape, dtype=gy.dtype)
        gx[..., self.mask] = gy
        return gx

class Take:
    def __init__(self, axis, indices):
        self.axis = axis
        self.indices = np.array(indices)

    def forward(self, x):
        self.x = x
        y = np.take(x, self.indices, axis=self.axis)
        return y

    def backward(self, gy):
        x = self.x
        axis = self.axis
        indices = self.indices.flatten()
        gx = np.zeros_like(x, dtype=Config.dtype)
        # 対称軸を平坦化
        #gy_trans = gy.reshape(*x.shape[:axis], -1, *x.shape[axis+1:])
        gy_trans = np.reshape(gy, (*x.shape[:axis], -1, *x.shape[axis+1:]))
        
        # 対象軸を一番前に持ってきながら、勾配をindicesの位置に設定
        gx = np.zeros_like(x, dtype=Config.dtype)
        np.add.at(np.moveaxis(gx, axis, 0),
                   indices,
                   np.moveaxis(gy_trans, axis, 0))
        return gx

class Permutations:
    " 多次元配列からaxisの指定する軸で順列を作る "
    def __init__(self, axis, n=None, r=2):
        self.axis = axis
        self.r = r
        self.take = None
        if n is not None:
            self.fix_configuration(n)

    def fix_configuration(self, n):    
        indices = itertools.permutations(range(n), self.r)
        self.take = Take(self.axis, np.array(list(indices)))

    def forward(self, x):
        if self.take is None:
            self.fix_configuration(x.shape[self.axis])
        return self.take.forward(x)

    def backward(self, gy):
        return self.take.backward(gy)

class Combinations:
    " 多次元配列からaxisの指定する軸で組合せを作る "
    def __init__(self, axis, n=None, r=2):
        self.axis = axis
        self.r = r
        self.take = None
        if n is not None:
            self.fix_configuration(n)

    def fix_configuration(self, n):    
        indices = itertools.combinations(range(n), self.r)
        self.take = Take(self.axis, np.array(list(indices)))

    def forward(self, x):
        if self.take is None:
            self.fix_configuration(x.shape[self.axis])
        return self.take.forward(x)

    def backward(self, gy):
        return self.take.backward(gy)

class TakePair:
    " 多次元配列からaxisの指定する軸で順列組合わせのペアを作る "
    def __init__(self, axis, method='permutation', n=None):
        self.axis = axis # 仮設定しfix_configurationで正規化して再設定　
        self.method = method
        self.take = None
        if n is not None:
            self.fix_configuration(n)

    def fix_configuration(self, shape):
        self.axis = self.axis % len(shape) # 軸の正規化
        n = shape[self.axis]
        if   self.method[:4] == 'perm':
            indices = itertools.permutations(range(n), 2)
        elif self.method[:4] == 'comb':
            indices = itertools.combinations(range(n), 2)
        else:
            raise Exception('Bad method.')
        indices = np.array(list(indices))
        self.take = Take(self.axis, indices)

    def forward(self, x):
        if self.take is None:
            self.fix_configuration(x.shape)
        y = self.take.forward(x)
        y = np.moveaxis(y, self.axis+1, 0) 
        return y[0], y[1]
    
    def backward(self, gy0, gy1):
        gy = np.stack([gy0, gy1], axis=self.axis+1)       
        gx = self.take.backward(gy)
        return gx


class TakePair2:
    " 多次元配列からaxisの指定する軸で順列組合わせのペアを作る "
    def __init__(self, axis, method='permutation', n=None):
        self.axis = axis # 仮設定しfix_configurationで正規化して再設定
        self.method = method
        self.take1 = None
        self.take2 = None
        if n is not None:
            self.fix_configuration(n)

    def fix_configuration(self, shape):    
        self.axis = self.axis % len(shape) # 軸の正規化
        n = shape[self.axis]
        if   self.method[:4] == 'perm':
            indices = itertools.permutations(range(n), 2)
        elif self.method[:4] == 'comb':
            indices = itertools.combinations(range(n), 2)
        else:
            raise Exception('Bad method.')
        indices = np.array(list(indices))
        self.take1 = Take(self.axis, indices[:, 0])
        self.take2 = Take(self.axis, indices[:, 1])

    def forward(self, x):
        if self.take1 is None or self.take2 is None:
            self.fix_configuration(x.shape)
        return self.take1.forward(x), self.take2.forward(x)

    def backward(self, gy1, gy2):
        gx1 = self.take1.backward(gy1)
        gx2 = self.take2.backward(gy2)
        return gx1 + gx2


############################################ 
# 以下、__forward__のみの定義 
############################################

class Step:
    def __init__(self, c=0):
        self.c = c
        
    def forward(self, x):
        return np.where(x <= self.c, 0, 1)

def step(x, c=0):
    return Step(c).forward(x)

class Equal:
    def forward(self, x0, x1):
        return x0 == x1

def equal(x0, x1):
    return Equal().forward(x0, x1)

class GreaterThan:
    def forward(self, x0, x1):
        return x0 > x1

def greater_than(x0, x1):
    return GreaterThan().forward(x0, x1)

class GreaterThanOrEqual:
    def forward(self, x0, x1):
        return x0 >= x1

def greater_than_or_equal(x0, x1):
    return GreaterThanOrEqual().forward(x0, x1)

class LessThan:
    def forward(self, x0, x1):
        return x0 < x1

def less_than(x0, x1):
    return LessThan().forward(x0, x1)

class LessThanOrEqual:
    def forward(self, x0, x1):
        return x0 <= x1

def less_than_or_equal(x0, x1):
    return LessThanOrEqual().forward(x0, x1)

class Argmax:
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        return np.argmax(x, axis=self.axis, keepdims=self.keepdims)

class Argmin:
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        return np.argmin(x, axis=self.axis, keepdims=self.keepdims)

class Argsort:
    def __init__(self, axis=None):
        self.axis = axis

    def forward(self, x):
        return np.argsort(x, axis=self.axis)

############################################ 
# 以下、Activatorsから仮移植 
############################################


class Sigmoid:
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y

    def backward(self, gy):
        y = self.y         
        gx = y * (1 - y) * gy
        return gx

class Tanh:
    def forward(self, x):
        y = np.tanh(x)
        self.y = y
        return y

    def backward(self, gy):
        y = self.y
        gx = gy * (1 - y * y)
        return gx

class Softmax:
    def forward(self, x):
        max_x = np.max(x, axis=-1, keepdims=True) #if dimx>1 else np.max(x)
        exp_a = np.exp(x - max_x)  # オーバーフロー対策
        sum_exp_a = np.sum(exp_a, axis=-1, keepdims=True) #if dimx>1 else np.sum(exp_a) 
        y = exp_a / (sum_exp_a + 1e-7)
        self.y = y
        return y

    def backward(self, gy): # ソフトマックス本来の逆伝播
        y = self.y
        gx = y * gy
        sumgx = np.sum(gx, axis=-1, keepdims=True)
        gx -= y * sumgx
        return gx

#######################################################
# HDArray OperatorOverload は nucleus 固有のため ufiesia では持たない
#######################################################

if __name__=='__main__':
    print('\n#### all cast ####')
    import inspect
    import sys
    current_module = sys.modules[__name__]
    classes = map(lambda x:x[0],inspect.getmembers(current_module,inspect.isclass))
    classes = list(classes)
    print(classes)
    
    #
    import matplotlib.pyplot as plt


    print('基本関数のテスト')
    #set_create_graph('True')

    #"""#
    print('オペランドが１つの関数')
    functions = (Assign, Neg, Abs, Sin, Cos, Square, Sqrt, Exp, Pow, Log, Erf)
    
    x = np.linspace(-4, 4)

    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)
        y = func.forward(x)
        gx = func.backward(np.ones_like(y))
        plt.plot(x.tolist(), y.tolist())
        plt.plot(x.tolist(), gx.tolist())
        plt.title(func.__class__.__name__)
        plt.show()

    print('オペランドが１つの関数 拡張')
    functions = (Exp, Pow, Log)
    
    x = np.linspace(-4, 4)
    a = 3.0

    for f in functions:
        func = f(a)
        print('test ', func.__class__.__name__)
        y = func.forward(x)
        gx = func.backward(np.ones_like(y))
        plt.plot(x.tolist(), y.tolist())
        plt.plot(x.tolist(), gx.tolist())
        plt.title(func.__class__.__name__)
        plt.show()


    print('オペランドが１つの関数 拡張2')
    functions = (Branch,)
    
    x = np.linspace(-4, 4)

    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)
        y = func.forward(x)
        gy1 = np.ones_like(y)
        gx = func.backward(gy1)
        plt.plot(x.tolist(), y.tolist())
        plt.plot(x.tolist(), gx.tolist())
        gy2 = np.ones_like(y)
        gx = func.backward(gy2, flush=False)
        plt.plot(x.tolist(), gx.tolist())
        plt.title(func.__class__.__name__)
        plt.show()


    print('オペランドが2つの関数')
    functions = (Add, Sub, Mul, Div)
    x0 = np.linspace(-4, 4) 
    x1 = np.linspace(4, -4)
    
    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)
        y = func.forward(x0, x1)
        gx0, gx1 = func.backward(np.ones_like(y))

        plt.plot(x0.tolist(), label='x0')
        plt.plot(x1.tolist(), label='x1')
        plt.plot(y.tolist(), label='y')
        plt.plot(gx0.tolist(), label='gx0')
        plt.plot(gx1.tolist(), label='gx1')
        plt.legend()
        plt.title(func.__class__.__name__)
        plt.show()

    print('オペランドが複数の関数')
    functions = (SumVariadic,)
    xs = []
    xs.append(np.linspace(0, 1))
    xs.append(np.linspace(1, 0))
    xs.append(np.linspace(-1, 1))
    #xs =(np.full((50,), i+1) for i in range(3))    
    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)
        y = func.forward(*xs)
        #print('xs =', xs)
        #print('y =', y)
        gxs = func.backward(np.ones_like(y))
        #print('gxs =', gxs)
        for i, x in enumerate(xs):
            plt.plot(x.tolist(), label='x'+str(i))
        plt.plot(y.tolist(), label='y')
        for i, gx in enumerate(gxs):
            plt.plot(gx.tolist(), label='gx'+str(i))
        plt.legend()
        plt.title(func.__class__.__name__)
        plt.show()

 
       
    #"""#
    #"""#
    print('基本関数の組み合わせのテスト')
    functions = (Normalize, L2Normalize)
    x = np.random.rand(10)

    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)

        y = func.forward(x)
        print(x.shape, y.shape)
        gy = np.arange(0, y.size) #np.random.rand(y.data.size)
        gy = gy[::-1]
        gx = func.backward(gy)

        print(type(x), type(y), type(gy), type(gx))

        plt.plot(x.tolist(), y.tolist())
        plt.plot(x.tolist(), gx.tolist())
        plt.plot(x.tolist(), gy.tolist())
        plt.grid()
        plt.title(func.__class__.__name__)
        plt.show()

    # define-by-run / nucleus 固有の backtrace テストは ufiesia では行わない。
    print('そのほかの関数のテスト')
    functions = (Transpose, Flatten, Max, Min)
    x = np.arange(12).reshape(3,4)
    print(x)
    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)
        y = func.forward(x)
        print(y)
        gx = func.backward(np.ones_like(y))
        print(gx)
    #"""#
    

    #'''#
    # CompositFunction / HDArray のテストは ufiesia では行わない。
    print('テンソル操作の関数のテスト')
    x = np.arange(24).reshape(2,3,4)
    a = (4,2,3)
    func = Reshape(a)
    print('test ', func.__class__.__name__, x.shape, '->', a)
    y = func.forward(x)  
    gx = func.backward(np.ones_like(y))
    print(x)
    print(y)
    print(gx)
    
    func = Reshape(4,2,3)
    print('test ', func.__class__.__name__, x.shape, '->', a)
    y = func.forward(x)  
    gx = func.backward(np.ones_like(y))
    print(x)
    print(y)
    print(gx)

    a = (2,0,1)
    func = Transpose(a)
    print('test ', func.__class__.__name__, x.shape, ':', a)
    y = func.forward(x)  
    gx = func.backward(np.ones_like(y))
    print(x)
    print(y)
    print(gx)

    func = Transpose(2,0,1)
    print('test ', func.__class__.__name__, x.shape, ':', a)
    y = func.forward(x)  
    gx = func.backward(np.ones_like(y))
    print(x)
    print(y)
    print(gx)

    x = x.reshape(6,4) 
    func = Transpose()
    print('test ', func.__class__.__name__, x.shape, ':', a)
    y = func.forward(x)  
    gx = func.backward(np.ones_like(y))
    print(x)
    print(y)
    print(gx)

    x0 = np.arange(12).reshape(3, 4)
    x1 = np.arange(12).reshape(4, 3)
    func = Dot()
    print('test ', func.__class__.__name__, x0.shape, x1.shape)
    y = func.forward(x0, x1)
    gx0, gx1 = func.backward(np.ones_like(y))
    
    print(x0)
    print(x1)
    print(y)
    print(gx0)
    print(gx1)
    
    x0 = np.arange(24).reshape(2, 3, 4)
    x1 = np.arange(24).reshape(2, 4, 3)
    func = MatMul()
    print('test ', func.__class__.__name__, x0.shape, x1.shape)
    y = func.forward(x0, x1)
    gx0, gx1 = func.backward(np.ones_like(y))
    
    print(x0)
    print(x1)
    print(y)
    print(gx0)
    print(gx1)
    
    #"""#
    print('そのほかの関数のテスト2')
    functions = (DotLinear, HadamardLinear, MatMulLinear)
    x = np.arange(4).reshape(2,2)
    w = np.arange(4).reshape(2,2)
    b = np.arange(2)
    print(x)
    print(w)
    print(b)
    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)
        y = func.forward(x, w, b)
        print(y)
        gx, gw, gb = func.backward(np.ones_like(y))
        print(gx)
        print(gw)
        print(gb)
    #"""#
    #"""#
    print('そのほかの関数のテスト3')
    functions = (MatMulLinear, MatMulLinear_bkup)
    x = np.arange(24).reshape(2,3,4)
    w = np.arange(8).reshape(4,2)
    b = np.arange(2)
    print(x)
    print(w)
    print(b)
    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)
        y = func.forward(x, w, b)
        print(y)
        gx, gw, gb = func.backward(np.ones_like(y))
        print(gx)
        print(gw)
        print(gb)
    #"""#
    #"""#
    print('そのほかの関数のテスト4')
    functions = (DualDotLinear,)
    x = np.arange(8).reshape(2,4)
    r = np.arange(6).reshape(2,3)
    w = np.arange(12).reshape(4,3)
    v = np.arange(9).reshape(3,3)
    b = np.arange(3)
    print(x)
    print(r)
    print(w)
    print(v)
    print(b)
    for f in functions:
        func = f()
        print('test ', func.__class__.__name__)
        y = func.forward(x, r, w, v, b)
        print(y)
        gx, gr, gw, gv, gb = func.backward(np.ones_like(y))
        print(gx)
        print(gr)
        print(gw)
        print(gv)
        print(gb)
    #"""#
    #"""#
    print('そのほかの関数のテスト3')
    functions = (ScaleDotLinear,)
    x = np.arange(24).reshape(2,3,4)
    w = np.arange(8).reshape(4,2)
    b = np.arange(2)
    g = np.array(2)
    print(x)
    print(w)
    print(b)
    print(g)
    for f in functions:
        func = f(scale=True)
        print('test ', func.__class__.__name__)
        y = func.forward(x, w, b, g)
        print(y)
        gx, gw, gb, gg = func.backward(np.ones_like(y))
        print(gx)
        print(gw)
        print(gb)
        print(gg)
    #"""#
    
    print('そのほかの関数のテスト4')
    func1 = Split(3)
    print('test ', func1.__class__.__name__)
    x = np.arange(3*2*4, dtype=np.float32).reshape(3,2,4)
    ys = func1.forward(x)
    print(ys)
    gys_in = tuple(np.ones_like(y) for y in ys)
    gx = func1.backward(*gys_in)
    print(gx)
    func2 = Concatenate()
    print('test ', func2.__class__.__name__)
    z = func2.forward(*ys)
    print(z)
    gys = func2.backward(np.ones_like(z))
    print(gys)


    print("reshape   =", np.reshape, getattr(np.reshape, "__module__", None))
    print("transpose =", np.transpose, getattr(np.transpose, "__module__", None))
    print("broadcast_to =", np.broadcast_to, getattr(np.broadcast_to, "__module__", None))
    print("expand_dims  =", np.expand_dims, getattr(np.expand_dims, "__module__", None))

