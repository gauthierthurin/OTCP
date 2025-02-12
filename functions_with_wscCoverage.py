import numpy as np
import ot


########################################################################################################################################
########################################################################################################################################
## CODES TO SOLVE OPTIMAL TRANSPORT / LEARN MK QUANTILES AND RANKS : 
########################################################################################################################################
########################################################################################################################################


def sample_grid(data,positive=False):
    n = data.shape[0]
    d = data.shape[1]
    R = np.linspace(0,1,n)
    # Reference distribution 
    if positive==False:
        mu = []
        for i in range(n):
            Z = np.random.normal(0,1,d)
            Z = Z / np.linalg.norm(Z)
            mu.append( R[i]*Z)
    else:
        mu = []
        for i in range(n):
            # 1 sphere
            Z = np.random.exponential(scale=1.0,size=d) 
            Z = Z / np.sum(Z)
            mu.append( R[i]*Z)
    return(np.array(mu))




def T0(x,DATA,psi): 
    ''' Returns the image of `x` by the OT map parameterized by `psi` towards the empirical distribution of `sample_sort`.'''
    if (len(x.shape)==1):  
        to_max = (DATA @ x) - psi 
        res = DATA[np.argmax(to_max)]
    else: 
        to_max = (DATA @ x.T).T - psi 
        res = DATA[np.argmax(to_max,axis=1)]
    return(res)  

def learn_psi(mu,data):
    M = ot.dist(data,mu)/2 
    res = ot.solve(M)
    g,f = res.potentials
    psi = 0.5*np.linalg.norm(mu,axis=1)**2 - f
    psi_star = 0.5*np.linalg.norm(data,axis=1)**2 - g 
    to_return = [psi,psi_star]
    return(to_return)

def RankFunc(x,mu,psi,ksi=0):
    # This computes a smooth argmax (LogSumExp). ksi is a regularisation parameter. Setting to 1000 removes ties in the data. 
    if (len(x.shape)==1):  
        to_max = ((mu @ x)- psi)*ksi
        to_sum = np.exp(to_max - np.max(to_max) )
        weights = to_sum/(np.sum(to_sum)) 
        res =  np.sum(mu*weights.reshape(len(weights),1),axis=0)
    else: 
        res=[]
        for xi in x:
            to_max = ((mu @ xi)- psi)*ksi
            to_sum = np.exp(to_max - np.max(to_max) )
            weights = to_sum/(np.sum(to_sum)) 
            res.append( np.sum(mu*weights.reshape(len(weights),1),axis=0))
        res = np.array(res)
    # For argsup, one can use T0. 
    if ksi == 0:
        res = T0(x,mu,psi) 
    return( res )

def QuantFunc(u,data,psi_star):
    return( T0(u,data,psi_star) )


def logsumexp(arg, axis, b):
    Max = np.max(arg)
    arg = np.exp(arg-Max)
    MAT = np.matmul(arg,b.reshape(-1,1))
    return(np.log(np.sum( MAT, axis=axis ))+Max)

def Sinkhome(alpha, beta, Cost, eps=1, n_sink=1000):
    I = len(alpha)
    J = len(beta)
    g = np.zeros(J)
    f = np.zeros(I)
    Cost = Cost/np.max(Cost)
   
    for k in range(n_sink):
        #update of the first dual variable
        g.shape = (1,-1)
        arg_one = (np.matmul(np.ones(shape = (I,1)),g) - Cost)/eps
        f = - eps*logsumexp(arg_one, axis=1, b=beta)

        #update of the second dual variable
        f.shape = (1,-1)
        arg_two = (np.matmul(np.ones(shape = (J,1)),f) - np.transpose(Cost))/eps
        g = - eps*logsumexp(arg_two, axis=1, b=alpha)

        # rescaling of the optimal dual variables
        m = np.mean(f)
        f = f-m
        g = g+m
    #Computation of the objective function    
    wass_approx = np.sum(f*alpha) + np.sum(g*beta) - eps
    return(f,g,wass_approx)

def MultivQuantileTreshold(data,alpha = 0.9,positive=False):
    ''' To change the reference distribution towards a positive one, set positive = True.  '''
    # Solve OT
    mu = sample_grid(data,positive=positive) 
    psi,psi_star = learn_psi(mu,data) 
    # Reference ranks (that are independent from the data)
    n = len(data)
    ReferenceRanks = np.linalg.norm(  mu ,axis=1,ord=2)  
    Quantile_Treshold = np.quantile( ReferenceRanks, np.min(  [np.ceil((n+1)*alpha)/n ,1] )   )
    ranksMK = None 
    Score_treshold = None 
    return(Quantile_Treshold,mu,psi,psi_star,ranksMK,Score_treshold) 




########################################################################################################################################
########################################################################################################################################
## CODES FOR REGRESSION : 
########################################################################################################################################
########################################################################################################################################



def get_volume_QR(Quantile_Treshold,mu,psi,scores,N = int(1e4)):
    """ Monte-Carlo estimation of the quantile region of order 'Quantile_Treshold'."""
    M = np.max(scores,axis=0)
    m = np.min(scores,axis=0)
    v = m + np.random.random((N,len(M)))*(M-m) 
    scale = np.prod(M-m)  
    MCMC = np.mean(np.linalg.norm( RankFunc(v,mu,psi) ,axis=1) <= Quantile_Treshold) 
    # This may be time-consuming, because we may sample lots of points for volume calculation
    return(MCMC*scale) 

def get_contourMK(Quantile_Treshold,psi_star,scores,N=100):
    contour = []
    angles = 2*np.pi*np.linspace(0,1,N)
    for theta in angles:
        us = np.array([[np.cos(theta)][0],[np.sin(theta)][0]])
        contour.append(us) 
    contour = np.array(contour) * Quantile_Treshold
    contourMK = QuantFunc(contour,scores,psi_star)
    return(contourMK)

from sklearn.neighbors import NearestNeighbors 

def ConformalQuantileReg(x,scores, n, x_tick,alpha):
    ''' 
    Return parameters related MK quantiles based on a quantile function that is conditional on x_tick. A neighborhood of x_tick is regarded within x, the calibration data. 

    - x = covariates of calibration data
    - scores = calibration scores, such as residuals, computed from predictions f(x) with same indices as in `x`
    - n= number of neighbors for KNN 
    - x_tick = a new point x where the conditional quantile function Q( . / X = x_tick) is to be computed 
    - alpha: confidence level in [0,1]
    ''' 

    x = np.array(x)

    knn = NearestNeighbors(n_neighbors=n)
    knn.fit(x)
    local_neighbors_test = knn.kneighbors(x_tick.reshape(1, -1), return_distance=False)
    indices_knn = local_neighbors_test.flatten()
    Y = scores[indices_knn][:n]  # Calibration scores associated to k nearest neighbors of x_tick (in x) 

    # The conformal quantile regions are paramterized by: (with twisted quantile level) 
    Quantile_Treshold,mu,psi,psi_star,ranksMK,Score_treshold = MultivQuantileTreshold(Y,alpha=alpha) 
    return(Quantile_Treshold, mu, psi,psi_star)






######################################################################
######################################################################
## CODES FOR CLASSIFICATION : 
######################################################################
######################################################################

def ScoreClassif(pi,BarY): 
    S = np.abs(BarY-pi)
    return(S)


####################################################
# OTCP 
####################################################

def func_prediction_set(pi_test,range_BarY,Quantile_Treshold,mu,psi):
    ''' Returns prediction set for our method, for classification.'''
    Prediction_Set = []
    for BarY in range_BarY:
        S_testy = ScoreClassif(pi_test,BarY)
        # Test if it is conform
        RankMK = RankFunc(S_testy, mu, psi) 
        norm_RankMK = np.linalg.norm(RankMK,axis=1,ord=2) 
        test = 1*(norm_RankMK <= Quantile_Treshold)
        # Gather results 
        Prediction_Set.append( test ) 
    Prediction_Set = np.array(Prediction_Set).T # multi-hot encoding 
    Prediction_Set = Prediction_Set *np.arange(1,pi_test.shape[1]+1) # replace ones by corresponding value of label
    Prediction_Set = [[i-1 for i in l if i != 0] for l in Prediction_Set.tolist()] 
    return(Prediction_Set)

from sklearn.preprocessing import LabelBinarizer # One hot encoding 
def calib_OTCP_classif(X_cal,y_cal,clf,alpha,K):
    enc = LabelBinarizer()
    range_BarY = enc.fit_transform(np.arange(K).reshape(K,1)) 
    BarY_cal = enc.transform(y_cal) 
    try:
        pi_cal = clf.predict_proba(X_cal)
    except:
        pi_cal = clf.predict(X_cal)
    S_cal = ScoreClassif(pi_cal,BarY_cal)
    Quantile_Treshold,mu,psi,psi_star,ranksMK,Score_treshold = MultivQuantileTreshold(S_cal,alpha=alpha,positive=True)
    L = [Quantile_Treshold,mu,psi,psi_star,clf,range_BarY]
    return( L )

def evaluate_OTCP_classif(Xtest,L):
    Quantile_Treshold,mu,psi,clf,range_BarY = L[0],L[1],L[2],L[4],L[5]
    try:
        pi_test = clf.predict_proba(Xtest)
    except:
        pi_test = clf.predict(Xtest)
    
    Prediction_Set = func_prediction_set(pi_test,range_BarY,Quantile_Treshold,mu,psi) 
    return(Prediction_Set  )


####################################################
# IP AND MS SCORES 
#################################################### 


def InverseProba(probas,y):
    '''
    Computes the Hinge Loss, with 'probas' of size (n,K) for n probabilities over K classes.
    y is the index of a class.
    '''
    return((1 - probas)[:,y])

def MarginScore(probas,y):
    '''
    Computes the Margin Score, with 'probas' of size (n,K) for n probabilities over K classes. 
    y is the index of a class.
    '''
    indexes = list(range( np.shape(probas)[1]  ))
    indexes.pop(y)
    MS = np.max(probas[:,indexes],axis=1) - probas[:,y]
    return(MS)


def calib_IP_MS_scores(pi_cal,y_cal,alpha):
    K = len(np.unique(y_cal))
    y = 0  # one iteration to initialize 
    IP_score = InverseProba(pi_cal[y_cal==y],y)
    MS_score = MarginScore(pi_cal[y_cal==y],y)
    for y in range(1,K): 
        s1 = InverseProba(pi_cal[y_cal==y],y)
        s2 = MarginScore(pi_cal[y_cal==y],y)
        IP_score = np.concatenate([IP_score, s1  ])
        MS_score = np.concatenate([MS_score, s2  ])
    IP_score = np.array(IP_score).T
    MS_score = np.array(MS_score).T 

    n = len(y_cal)
    q = alpha * (1+1/n)
    Q1 = np.quantile(IP_score,q) 
    Q2 = np.quantile(MS_score,q)
    return (Q1,Q2)


def evaluate_IP_MS_scores(pi_test,Q1,Q2,K):
    Prediction_Set_IP = [] 
    Prediction_Set_MS = []
    for y in np.arange(K): 
        test = (InverseProba(pi_test,y) <= Q1)
        Prediction_Set_IP.append( test ) 
        test = (MarginScore(pi_test,y) <= Q2 ) 
        Prediction_Set_MS.append(test ) 
    Prediction_Set_MS = np.array(Prediction_Set_MS).T # multi-hot encoding 
    Prediction_Set_IP = np.array(Prediction_Set_IP).T # multi-hot encoding 
    Prediction_Set_MS = Prediction_Set_MS *np.arange(1,pi_test.shape[1]+1) # replace ones by corresponding value of label 
    Prediction_Set_MS = [[i-1 for i in l if i != 0] for l in Prediction_Set_MS.tolist()] 

    Prediction_Set_IP = Prediction_Set_IP *np.arange(1,pi_test.shape[1]+1) # replace ones by corresponding value of label
    Prediction_Set_IP = [[i-1 for i in l if i != 0] for l in Prediction_Set_IP.tolist()] 
    return( Prediction_Set_IP,Prediction_Set_MS )


####################################################
# GET METRICS FROM PREDICTION SET
#################################################### 


# arc contains codes for APS taken from : # https://sites.google.com/view/cqr/classification-example-1?authuser=0
from arc import coverage 
from arc import methods 
# Important: the file methods has been changed so that methods.SplitConformal only performs calibration. This is specified in methods.py
####################################################

def get_metrics(predictions,y,X):
    MarginalCoverage = np.mean([y[i] in predictions[i] for i in range(len(y))])
    Efficiency = np.mean([len(predictions[i]) for i in range(len(y))]) # size of prediction set 
    Informativeness = np.mean([1*(len(predictions[i]) == 1) for i in range(len(y))]) # proportion of singletons
    wsc_coverage = coverage.wsc_unbiased(X, y, predictions,verbose=True,delta=0.1,M=1000) # verbose = True to show bar indicating time left for computation
    return(MarginalCoverage, Efficiency, Informativeness,wsc_coverage) 


def Metrics_AllMethods(X_test,y_test,X_cal,y_cal,clf, alpha,K):
    ''' Calibrate all methods on X_cal,y_cal and test on X_test,y_test '''
    # Apply OTCP
    L = calib_OTCP_classif(X_cal,y_cal,clf,alpha,K)
    Prediction_Set = evaluate_OTCP_classif(X_test,L)  
    res_OTCP  = get_metrics(Prediction_Set,y_test,X_test) 

    try:
        pi_cal = clf.predict_proba(X_cal)
        pi_test = clf.predict_proba(X_test)
    except:
        pi_cal = clf.predict(X_cal)
        pi_test = clf.predict(X_test)

    # Apply IP and MS 
    Q1,Q2 = calib_IP_MS_scores(pi_cal,y_cal,alpha)

    Prediction_Set_IP,Prediction_Set_MS = evaluate_IP_MS_scores(pi_test,Q1,Q2,len(np.unique(y_cal)))
    res_IP = get_metrics(Prediction_Set_IP,y_test,X_test) 
    res_MS = get_metrics(Prediction_Set_MS,y_test,X_test) 

    # Apply APS 
    method_sc = methods.SplitConformal(X_cal, y_cal, clf, 1-alpha)
    Prediction_Set_ARS = method_sc.predict(X_test)
    res_APS = get_metrics(Prediction_Set_ARS, y_test,X_test) 
    return(res_OTCP,res_IP,res_MS, res_APS)

def CalibAllMetrics(X_cal,y_cal,clf, alpha,K):
    '''Calibrate all methods on X_cal,y_cal '''

    # APS 
    method_sc = methods.SplitConformal(X_cal, y_cal, clf, 1-alpha) # Calibration on X_cal and y_cal 

    try:
        pi_cal = clf.predict_proba(X_cal)
    except:
        pi_cal = clf.predict(X_cal)
    # OTCP
    enc = LabelBinarizer()
    range_BarY = enc.fit_transform(np.arange(K).reshape(K,1)) 
    BarY_cal = enc.transform(y_cal) 
    S_cal = ScoreClassif(pi_cal,BarY_cal)

    Quantile_Treshold,mu,psi,psi_star,ranksMK,Score_treshold = MultivQuantileTreshold(S_cal,alpha=alpha,positive=True)
    L = [Quantile_Treshold,mu,psi,psi_star,clf,range_BarY]

    # IP and MS 
    Q1,Q2 = calib_IP_MS_scores(pi_cal,y_cal,alpha)

    calib_parameters = [L,Q1,Q2,method_sc]
    return(calib_parameters)

def TestAllMetrics(X_test,y_test,clf,calib_parameters,K):
    ''' Test all methods on X_test,y_test '''
    try:
        pi_test = clf.predict_proba(X_test)
    except:
        pi_test = clf.predict(X_test)

    L,Q1,Q2,method_sc = calib_parameters
    # OTCP
    Quantile_Treshold,mu,psi,range_BarY = L[0],L[1],L[2],L[5]
    Prediction_Set = func_prediction_set(pi_test,range_BarY,Quantile_Treshold,mu,psi) 
    res_OTCP  = get_metrics(Prediction_Set,y_test,X_test) 
    # IP and MS 
    Prediction_Set_IP,Prediction_Set_MS = evaluate_IP_MS_scores(pi_test,Q1,Q2,K)
    res_IP = get_metrics(Prediction_Set_IP,y_test,X_test) 
    res_MS = get_metrics(Prediction_Set_MS,y_test,X_test)
    # APS 
    Prediction_Set_ARS = method_sc.predict(X_test)
    res_APS = get_metrics(Prediction_Set_ARS, y_test,X_test)
    return(res_OTCP,res_IP,res_MS, res_APS)





















