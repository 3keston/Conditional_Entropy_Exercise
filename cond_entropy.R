# conditional entropy for X_1,...X_n
# refs: 
# Machine Learning, Tom Mitchell, McGraw Hill, 1997, pg 55
# http://www.autonlab.org/tutorials/infogain.html

# data
x = c(rep(1, 2), rep(0, 2))
y = c(rep(1, 1), rep(0, 3))

yi = c(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)

x0 = c(0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0)
x1 = c(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0)
x2 = c(0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0)
x3 = c(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0)

x4 = c(1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0)
x5 = c(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0)
x6 = rep(0, 16)
x7 = rep(1, 16)

ds = data.frame(X=c(0,1,2,0,0,2,1,0), Y=c(1,0,1,0,0,1,0,1))

df = data.frame(yi, 
                x0,x1,x2,x3,x4,x5,x6,x7)

# functions
make.joint = function(df) {
  #
  # H(X_{1},...,X_{n}) =
  # \sum_{x_{1},...,x_{n}p(x_{1},...,x_{n})log2(p(x_{1},...,x_{n}))
  # log2 is used because the expected encoding length is measured in bits

  pt = table(df) # joint frequency table
  pt = pt / sum(pt) # joint prob table
  pt = pt[pt > 0] # pt*log2(pt) is 0 if p(x_{1}...x_{n}) = 0
  
  H = -sum(pt*log2(pt))
  return(H)
}

make.condi = function(df) {
  # assumes the first column is Y where H(Y|X)
  # uses the chain rule for conditional entropy
  # H(Y|X) = H(X,Y) - H(X)

  H.con = make.joint(df) - make.joint(df[,2:ncol(df)])
  return(H.con)
}

make.ig = function(df, rela=FALSE) {
  #          IG(Y|X) = H(Y) - H(Y|X)
  # relative IG(Y|X) = H(Y) - H(Y|X) / H(Y)

  HY = make.joint(df[,1])
  IG = HY - make.condi(df)

  if(rela==TRUE) {
    IG = IG / HY
  } 
  return(IG)
}

# TODO
# add split information and IG ratio, which is gain / splitinfo
# pg 74 Mitchell
# https://en.wikipedia.org/wiki/Information_gain_ratio
# Quinlan 1986

#val labels = Array(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)

# ex0
#assert(jointEntropy(Array.fill(16)(0),
#                    Array.fill(16)(0),
#                    labels) 
#                    == 1.0)
# ex1
#assert(jointEntropy(Array.fill(16)(0),
#                    labels,
#                    labels) 
#                    == 0.0)
# ex2
#assert(jointEntropy(Array.fill(16)(1), 
#                    Array(0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0),
#                    labels) 
#                    == 1.0)
# ex3
#assert(jointEntropy(Array.fill(16)(0),
#                    Array(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
#                    labels) 
#                    == 1.0)
# ex4
#assert(jointEntropy(Array(0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0),
#                    Array(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
#                    labels) 
#                    == 0.0)
# ex5
#val entropy1 = jointEntropy(Array(1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1,0), 
#                            Array(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0), 
#                            labels) assert(math.abs(entropy1 - 0.344) < 0.01)

ex0 = df[,c("yi","x6","x6")] # conditional on all zeros
make.condi(ex0)

ex1 = df[,c("yi","x6","yi")] # conditional on zeros and labels
make.condi(ex1)

ex2 = df[,c("yi","x7","x0")] # 1s then Array(0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1,
#                                             0, 1, 0, 1, 0)
make.condi(ex2)

ex3 = df[,c("yi","x6","x1")] # zeros then Array(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
#                                            1, 0, 1, 0)
make.condi(ex3)

ex4 = df[,c("yi","x0","x1")] # (Array(0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0),
#                               Array(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,0))
make.condi(ex4)

ex5 = df[,c("yi","x4","x5")] # Array(0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0),
#                              Array(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0)
abs(make.condi(ex5) - 0.344) < 0.01
# should be true

# higher dimensional example
ex6 = df[,c("yi","x1","x5","x0")]
make.condi(ex6)

#
