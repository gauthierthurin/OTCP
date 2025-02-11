import numpy as np
import ot


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
        # UNIFORM-LIKE ON 2-SPHERE OR 1-SPHERE:
        mu = []
        for i in range(n):
            # 2 sphere
            #Z = np.abs(np.random.normal(0,1,d))
            #Z = Z / np.linalg.norm(Z)
            # 1 sphere
            Z = np.random.exponential(scale=1.0,size=d) 
            Z = Z / np.sum(Z)
            mu.append( R[i]*Z)
        # HYPERCUBE:
        #mu = np.random.random(data.shape)
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

def OT_exact_(U,Y):
    ''' Uses ot.solve on scaled data, before rescaling back'''
    M = ot.dist(U,Y)/2
    res = ot.solve(M)
    f,g = res.potentials
    psi_star = 0.5*np.linalg.norm(Y,axis=1)**2 - g
    return(psi_star) 

#- tester un classifieur random 
#- dans quelles situations le fait de prendre en compte 3(au lieu de 2 ) probas est mieux ? 
#Hypotheses: conditional coverage, et si le classifieur se trompe totalement 
#( une classe = une gaussienne petite, l'autre = 2 gaussiennes de part et d'autre, 3e classe random )

def learn_psi(mu,data):
    psi_star = OT_exact_(mu,data) 
    psi = [] 
    for u in mu:
        to_max = (data @ u) - psi_star
        psi.append( np.max(to_max) )
    psi = np.array(psi)
    return([psi,psi_star])

def learn_psi(mu,data):
    M = ot.dist(data,mu)/2 
    res = ot.solve(M)
    g,f = res.potentials
    psi = 0.5*np.linalg.norm(mu,axis=1)**2 - f
    psi_star = 0.5*np.linalg.norm(data,axis=1)**2 - g 
    #psi_star = [] 
    #for x in data:
    #    to_max = (mu @ x) - psi
    #    psi_star.append( np.max(to_max) )
    #psi_star = np.array(psi_star)
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
   
    # Initilisation du temps
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


#def func_prediction_set(pi_test,range_BarY,Quantile_Treshold,mu,psi):
#    ''' Returns prediction set for our method, for classification.'''
#    Prediction_Set = []
#    for y in range_BarY:
#        S_testy = np.abs(y-pi_test)
        #toto = S_testy + 1e-8*np.random.random(S_testy.shape)
        #toto = np.log(toto)
        #S_testy = toto 
#        # Test if it is conform
#        RankMK = RankFunc(S_testy, mu, psi) 
#        norm_RankMK = np.linalg.norm(RankMK,axis=1,ord=2)  # CAREFUL GAUTHIER
#        test = 1*(norm_RankMK <= Quantile_Treshold)
#        # Gather results 
#        Prediction_Set.append( test ) 
#    Prediction_Set = np.array(Prediction_Set).T # multi-hot encoding 
#    Prediction_Set = Prediction_Set *np.arange(1,pi_test.shape[1]+1) # replace ones by corresponding value of label
#    Prediction_Set = [[i-1 for i in l if i != 0] for l in Prediction_Set.tolist()] 
#    return(Prediction_Set)

#def get_metrics(predictions,test_y):
#    MarginalCoverage = 0 
#    Efficiency = 0 #average size of prediction set 
#    Informativeness = 0 #percentage of predicted sets of size 1 
#    n = len(predictions)
#    for j in range(n):
#        MarginalCoverage += (test_y[j] in predictions[j])
#        Efficiency += len(predictions[j])
#        Informativeness += 1*(len(predictions[j]) == 1)
#
#    MarginalCoverage = MarginalCoverage / n
#    Efficiency = Efficiency / n
#    Informativeness = Informativeness / n 
#    return(MarginalCoverage, Efficiency, Informativeness )

def MultivQuantileTreshold(data,alpha = 0.9,positive=False):
    ''' To change the reference distribution towards a positive one, set positive = True.  '''
    # Solve OT
    # np.random.seed(62) # to limit additionnal randomness inherent to MK multivariate ranks 
    mu = sample_grid(data,positive=positive) 

    #psi,psi_star = learn_psi(mu,data) # Once psi is learnt, one can compute the rank function in linear time with respect to n. 
    psi,psi_star = learn_psi(mu,data) 
    # TO COUNT IF THERE IS TIES:
    #VectorRanks =  RankFunc(data,mu,psi)
    #ranksMK = np.linalg.norm(  VectorRanks ,axis=1 ) 
    #n = len(data) 
    #Quantile_Treshold = np.ceil((n+1)*alpha) / n 
    #print( Quantile_Treshold )
    #Score_treshold = None 
    ####################################################################################
    # ANCIENNE VERSION, DEBUT 
    ############ 
    ####  Change n so that it equals the number of different ranks. This has the drawback of reducing n and lowering theoretical confidence. 
    #idx = np.unique(np.sort(ranksMK),return_index=True)[1] # 
    #OrderStatistics = np.argsort(ranksMK)[idx] # Keeps only the ranks that are different, i.e. remove redundancies 
    #n = len(OrderStatistics) # Count the number of different ranks, that can be lower than len(scores) due to imprecise calculus
    #print("ties?",len(OrderStatistics),len(data) )
    ####################################################################################
    # Compute quantile treshold 
    #q = alpha *(1+1/n) 
    #if (np.ceil(q*n))<=n: 
    #    indexTreshold = OrderStatistics[int(q*n)] #Here, int instead of np.ceil because counting in python begins at 0 !  
    #else: 
    #    indexTreshold = OrderStatistics[n-1] 
    #Score_treshold = data[indexTreshold] 
    #Quantile_Treshold = np.linalg.norm(  RankFunc(Score_treshold,mu,psi) )  # Under the norm of this limit : inlier. Otherwise : outlier.
    # ANCIENNE VERSION : FIN 
    ####################################################################################
    # CALCULER QUANTILE TRESHOLDS A PARTIR DE LA MATRICE DE TO 
    n = len(data)
    ReferenceRanks = np.linalg.norm(  mu ,axis=1,ord=2)  
    #print(ReferenceRanks.shape) 
    Quantile_Treshold = np.quantile( ReferenceRanks, np.min(  [np.ceil((n+1)*alpha)/n ,1] )   )
    #print( Quantile_Treshold )
    ranksMK = None 
    Score_treshold = None 
    ####################################################################################
    #n = len(data)
    #ranksMK = RankFunc(data,mu,psi)
    #normRanksMK = np.linalg.norm(  ranksMK ,axis=1,ord=2)  
    ##n = len(np.unique( normRanksMK ) ) # to tackle ties 
    #Quantile_Treshold = np.quantile( normRanksMK, np.min(  [np.ceil((n+1)*alpha)/n ,1] )   )
    #Score_treshold = None 
    return(Quantile_Treshold,mu,psi,psi_star,ranksMK,Score_treshold) 

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




##############################################################################
##############################################################################


def QuantileKn_d(x, y, n, x_tick,k,levels):
    ### x=covariates 
    ### y=output variables
    ### n= number of neighbors 
    ### x_tick = list of points x where the conditional quantile function Q( . / X = x) is to be computed 
    ### k number of points in quantile contour
    ### levels : contour levels to be computed, each between 0 and 1 

    x = np.array(x)
    y = np.array(y)    

    ## uniform grid
    d = y.shape[1]
    U = sample_grid(np.zeros((n,d)))

    ## Compute quantile contours, for each x
    for i in range(len(x_tick)):
        X = np.zeros(n)
        ### select n-nearest neighbors
        order = np.argsort(np.linalg.norm(x-x_tick[i],axis=1) )
        X = x[order][:n]
        Y = y[order][:n]

        psi_star = OT_exact_(U,Y) ## POT with scaling/rescaling  
        
        quantile_contours = []
        for alpha in levels:
            sphere = np.random.multivariate_normal(np.zeros(d),np.eye(d),k)
            sphere = sphere / np.linalg.norm(sphere,axis=1).reshape((k,1))
            quantile_contours.append( T0(alpha*sphere , Y ,psi_star) )
        quantile_contours = np.array(quantile_contours)
    return(quantile_contours)
        



##############################################################################
##############################################################################
from math import inf
def OT_exact(U,Y):
    ''' Compute the simplex algorithm, and OT potentials and maps. `U` and `Y` shall contain the same number of points. Taken from  https://github.com/monpCOqr/Nonparametric-multiple-output-center-outward-quantile-regression/tree/main '''
    n = U.shape[0]
    d = U.shape[1]
    ## computation of cost matrix
    cost = ot.dist(U,Y)
    ## optimal transport
    w = np.ones(n)
    Pn, log = ot.emd(w, w, cost, log=True, numItermax=100000000)
    new_index = (Pn @ np.arange(n)).astype('int32')
    sample_sort = Y[new_index]

    ## renormalization 
    ysup = max(np.linalg.norm(sample_sort, axis=1))
    xxx = U / ysup
    yyy = sample_sort / ysup 

    ## Karp's algorithm for computing the weights
    xiyi = (xxx * yyy) @ np.ones(shape=(d, 1))
    cij = xiyi @ np.ones(shape=(1, n)) - xxx @ yyy.T
    for u in range(n):
        cij[u,u] = inf

    dij = np.zeros(shape=(n + 1, n))
    dij[0, :] = inf
    dij[0, 0] = 0
    for kk in range(1, n + 1):
        for u in range(n):
            dij[kk, u] = min(dij[(kk - 1), :] + cij[:, u])

    dndk = np.zeros(n)
    denom = (np.zeros(n) + n) - list(range(n))

    for u in range(n):
        dndk[u] = max((dij[n, u] - dij[:-1, u]) / denom)
    e_star = min(dndk)/2

    mat1 = np.arange(0, n + 1)
    mat1.shape = (n + 1, 1)
    mat2 = np.ones(n)
    mat2.shape = (1, n)

    di_tilde = (dij - e_star * mat1 @ mat2).min(axis=0)
    psi = -di_tilde * ysup ** 2
    e0 = (abs(e_star) * ysup ** 2 / 2)/2  

    return(psi,sample_sort)






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

#def get_metrics(predictions,y,X):
#    MarginalCoverage = np.mean([y[i] in predictions[i] for i in range(len(y))])
#    Efficiency = np.mean([len(predictions[i]) for i in range(len(y))]) # size of prediction set 
#    Informativeness = np.mean([1*(len(predictions[i]) == 1) for i in range(len(y))]) # proportion of singletons
#    wsc_coverage = coverage.wsc_unbiased(X, y, predictions,verbose=True,delta=0.1,M=200) # verbose = True to show bar indicating time left for computation
#    return(MarginalCoverage, Efficiency, Informativeness,wsc_coverage) 

def get_metrics(predictions,y,X):
    ''' Computes coverage, efficiency and informativeness. 
    Due to the time required, we do not compute WSC coverage. It can be computed by removing comments above. '''
    MarginalCoverage = np.mean([y[i] in predictions[i] for i in range(len(y))])
    Efficiency = np.mean([len(predictions[i]) for i in range(len(y))]) # size of prediction set 
    Informativeness = np.mean([1*(len(predictions[i]) == 1) for i in range(len(y))]) # proportion of singletons
    wsc_coverage = -1 
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
    method_sc = methods.SplitConformal(X_cal, y_cal, clf, 1-alpha)

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





















