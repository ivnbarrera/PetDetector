from fastai.vision.all import tensor, TensorPoint, Transform, torch, L, params

def img2kpts(f):
  fname = str(f).split("/")[-1]
  try:
    return df_train.loc[df_train.filename==fname, kpt_cols].values[0]
  except IndexError as e:
    raise RuntimeError(fname + ' not in annotations') from e

def sep_points(coords):
  "Seperate a set of points to groups"
  kpts = []
  for i in range(1, int(len(coords/2)), 2):
    kpts.append([coords[i-1], coords[i]])
  return tensor(kpts)

def get_y(f):
  "Get keypoints for `f` image"
  pts = img2kpts(f)
  return sep_points(pts)

def get_ip(img, pts): return TensorPoint(pts, sz=img.size)

class ClampBatch(Transform):
  "Clamp points to a minimum and maximum in a batch"
  order = 4
  def __init__(self, min=-1, max=1, **kwargs):
    super().__init__(**kwargs)
    self.min, self.max = min, max
  def encodes(self, x:(TensorPoint)):
    for i, sets in enumerate(x):
      for j, pair in enumerate(sets):
        cpnt = torch.clamp(pair, self.min, self.max)
        if any(cpnt>=1) or any(cpnt<=-1):
          x[i][j] = tensor([-1,-1])
    return x

def _resnet_split(m): return L(m[0][:6], m[0][6:], m[1:]).map(params)