Search.setIndex({envversion:50,filenames:["gptf","gptf.core","index","modules"],objects:{"":{gptf:[0,0,0,"-"]},"gptf.core":{densities:[1,0,0,"-"],kernels:[1,0,0,"-"],likelihoods:[1,0,0,"-"],meanfunctions:[1,0,0,"-"],models:[1,0,0,"-"],params:[1,0,0,"-"],tfhacks:[1,0,0,"-"],transforms:[1,0,0,"-"],trees:[1,0,0,"-"],utils:[1,0,0,"-"],wrappedtf:[1,0,0,"-"]},"gptf.core.densities":{bernoulli:[1,1,1,""],beta:[1,1,1,""],exponential:[1,1,1,""],gamma:[1,1,1,""],gaussian:[1,1,1,""],laplace:[1,1,1,""],lognormal:[1,1,1,""],multivariate_normal:[1,1,1,""],poisson:[1,1,1,""],student_t:[1,1,1,""]},"gptf.core.kernels":{Absolute:[1,2,1,""],Additive:[1,2,1,""],Bias:[1,4,1,""],Constant:[1,2,1,""],Cosine:[1,2,1,""],Divisive:[1,2,1,""],Exponential:[1,2,1,""],Kernel:[1,2,1,""],Matern12:[1,2,1,""],Matern32:[1,2,1,""],Matern52:[1,2,1,""],Multiplicative:[1,2,1,""],Negative:[1,2,1,""],PartiallyActive:[1,2,1,""],RBF:[1,2,1,""],Static:[1,2,1,""],Stationary:[1,2,1,""],White:[1,2,1,""]},"gptf.core.kernels.Absolute":{K:[1,3,1,""],Kdiag:[1,3,1,""]},"gptf.core.kernels.Additive":{K:[1,3,1,""],Kdiag:[1,3,1,""]},"gptf.core.kernels.Constant":{K:[1,3,1,""]},"gptf.core.kernels.Divisive":{K:[1,3,1,""],Kdiag:[1,3,1,""]},"gptf.core.kernels.Kernel":{K:[1,3,1,""],Kdiag:[1,3,1,""],compute_K:[1,3,1,""],compute_K_symm:[1,3,1,""],compute_Kdiag:[1,3,1,""]},"gptf.core.kernels.Multiplicative":{K:[1,3,1,""],Kdiag:[1,3,1,""]},"gptf.core.kernels.Negative":{K:[1,3,1,""],Kdiag:[1,3,1,""]},"gptf.core.kernels.PartiallyActive":{K:[1,3,1,""],Kdiag:[1,3,1,""],active_dims:[1,4,1,""],wrapped:[1,4,1,""]},"gptf.core.kernels.Static":{Kdiag:[1,3,1,""],variance:[1,4,1,""]},"gptf.core.kernels.Stationary":{K:[1,3,1,""],Kdiag:[1,3,1,""],euclid_dist:[1,3,1,""],lengthscales:[1,4,1,""],square_dist:[1,3,1,""],variance:[1,4,1,""]},"gptf.core.kernels.White":{K:[1,3,1,""]},"gptf.core.likelihoods":{Gaussian:[1,2,1,""],Likelihood:[1,2,1,""]},"gptf.core.likelihoods.Gaussian":{conditional_mean:[1,3,1,""],conditional_variance:[1,3,1,""],logp:[1,3,1,""],predict_density:[1,3,1,""],predict_mean_and_var:[1,3,1,""],variational_expectations:[1,3,1,""]},"gptf.core.likelihoods.Likelihood":{conditional_mean:[1,3,1,""],conditional_variance:[1,3,1,""],logp:[1,3,1,""],predict_density:[1,3,1,""],predict_mean_and_var:[1,3,1,""],variational_expectations:[1,3,1,""]},"gptf.core.meanfunctions":{Absolute:[1,2,1,""],Additive:[1,2,1,""],Constant:[1,2,1,""],Divisive:[1,2,1,""],Linear:[1,2,1,""],MeanFunction:[1,2,1,""],Multiplicative:[1,2,1,""],Negative:[1,2,1,""],Zero:[1,2,1,""]},"gptf.core.meanfunctions.Divisive":{clone:[1,3,1,""]},"gptf.core.models":{GPModel:[1,2,1,""],Model:[1,2,1,""]},"gptf.core.models.GPModel":{build_posterior_mean_var:[1,3,1,""],build_prior_mean_var:[1,3,1,""],compute_posterior_mean_cov:[1,3,1,""],compute_posterior_mean_var:[1,3,1,""],compute_posterior_samples:[1,3,1,""],compute_prior_mean_cov:[1,3,1,""],compute_prior_mean_var:[1,3,1,""],compute_prior_samples:[1,3,1,""],predict_density:[1,3,1,""],predict_y:[1,3,1,""]},"gptf.core.models.Model":{build_log_likelihood:[1,3,1,""],build_log_prior:[1,3,1,""],compute_log_likelihood:[1,3,1,""],compute_log_prior:[1,3,1,""],optimize:[1,3,1,""]},"gptf.core.params":{DTypeChangeError:[1,5,1,""],DataHolder:[1,2,1,""],FixedParameterError:[1,5,1,""],Param:[1,2,1,""],ParamAttributes:[1,2,1,""],ParamList:[1,2,1,""],Parameterised:[1,4,1,""],Parameterized:[1,2,1,""],Proxy:[1,2,1,""],ProxyWrappedValue:[1,2,1,""],ShapeChangeError:[1,5,1,""],WrappedValue:[1,2,1,""],autoflow:[1,1,1,""],no_gc:[1,1,1,""],share_properties:[1,1,1,""]},"gptf.core.params.DataHolder":{feed_dict:[1,4,1,""],on_shape_change:[1,4,1,""],tensor:[1,4,1,""],value:[1,4,1,""]},"gptf.core.params.Param":{clear_cache:[1,3,1,""],feed_dict:[1,4,1,""],fixed:[1,4,1,""],free_state:[1,4,1,""],initializer:[1,4,1,""],on_session_birth:[1,3,1,""],on_session_death:[1,3,1,""],tensor:[1,4,1,""],transform:[1,4,1,""],value:[1,4,1,""]},"gptf.core.params.Parameterized":{ARRAY_DISPLAY_LENGTH:[1,4,1,""],data_holders:[1,4,1,""],data_summary:[1,3,1,""],feed_dict:[1,4,1,""],fixed:[1,4,1,""],param_summary:[1,3,1,""],params:[1,4,1,""],summary:[1,3,1,""]},"gptf.core.params.Proxy":{Shared:[1,2,1,""],copy:[1,3,1,""]},"gptf.core.params.ProxyWrappedValue":{cache:[1,4,1,""],clear_all_ancestor_caches:[1,3,1,""],on_dtype_change:[1,4,1,""],on_shape_change:[1,4,1,""],tf_device:[1,4,1,""],tf_graph:[1,4,1,""],tf_session_target:[1,4,1,""]},"gptf.core.params.WrappedValue":{on_dtype_change:[1,4,1,""],on_shape_change:[1,4,1,""],value:[1,4,1,""]},"gptf.core.tfhacks":{eye:[1,1,1,""]},"gptf.core.transforms":{Exp:[1,2,1,""],Identity:[1,2,1,""],Transform:[1,2,1,""]},"gptf.core.transforms.Exp":{np_backward:[1,3,1,""],tf_forward:[1,3,1,""]},"gptf.core.transforms.Identity":{np_backward:[1,6,1,""],tf_forward:[1,6,1,""]},"gptf.core.transforms.Transform":{np_backward:[1,3,1,""],tf_forward:[1,3,1,""]},"gptf.core.trees":{AttributeTree:[1,2,1,""],BadParentError:[1,5,1,""],BreadthFirstTreeIterator:[1,2,1,""],Leaf:[1,2,1,""],ListTree:[1,2,1,""],Tree:[1,2,1,""],TreeWithCache:[1,2,1,""],cache_method:[1,1,1,""]},"gptf.core.trees.AttributeTree":{children:[1,4,1,""],copy:[1,3,1,""],fallback_name:[1,4,1,""],name_of:[1,3,1,""]},"gptf.core.trees.Leaf":{children:[1,4,1,""],copy:[1,3,1,""]},"gptf.core.trees.ListTree":{children:[1,4,1,""],copy:[1,3,1,""],insert:[1,3,1,""],name_of:[1,3,1,""]},"gptf.core.trees.Tree":{children:[1,4,1,""],copy:[1,3,1,""],highest_parent:[1,4,1,""],long_name:[1,4,1,""],name:[1,4,1,""],name_of:[1,3,1,""],parent:[1,4,1,""]},"gptf.core.trees.TreeWithCache":{cache:[1,4,1,""],clear_ancestor_caches:[1,3,1,""],clear_cache:[1,3,1,""],clear_subtree_caches:[1,3,1,""],clear_tree_caches:[1,3,1,""],copy:[1,3,1,""]},"gptf.core.utils":{LRUCache:[1,2,1,""],combine_fancy_tables:[1,1,1,""],construct_table:[1,1,1,""],flip:[1,1,1,""],is_array_like:[1,1,1,""],isattrof:[1,1,1,""],isclassof:[1,1,1,""],prefix_lines:[1,1,1,""],strip_ansi_escape_codes:[1,1,1,""],unique:[1,1,1,""]},"gptf.core.wrappedtf":{NullContextWrapper:[1,2,1,""],WrappedTF:[1,2,1,""],WrappedTFSession:[1,2,1,""],tf_method:[1,1,1,""]},"gptf.core.wrappedtf.NullContextWrapper":{_NullContextWrapped__wrapped:[1,4,1,""]},"gptf.core.wrappedtf.WrappedTF":{NO_DEVICE:[1,4,1,""],copy:[1,3,1,""],get_session:[1,3,1,""],on_session_birth:[1,3,1,""],on_session_death:[1,3,1,""],op_placement_context:[1,3,1,""],tf_device:[1,4,1,""],tf_graph:[1,4,1,""],tf_session_target:[1,4,1,""]},"gptf.core.wrappedtf.WrappedTFSession":{close:[1,3,1,""]},"gptf.distributed":{BCMReduction:[0,2,1,""],PoEReduction:[0,2,1,""],PriorDivisorReduction:[0,2,1,""],Reduction:[0,2,1,""],cao_fleet_weights:[0,1,1,""],chunks:[0,1,1,""],distributed_tree_rBCM:[0,1,1,""],equal_weights:[0,1,1,""],gPoEReduction:[0,2,1,""],ones_weights:[0,1,1,""],rBCMReduction:[0,2,1,""],tree_rBCM:[0,1,1,""]},"gptf.distributed.BCMReduction":{build_posterior_mean_var:[0,3,1,""]},"gptf.distributed.PoEReduction":{build_posterior_mean_var:[0,3,1,""]},"gptf.distributed.PriorDivisorReduction":{build_log_likelihood:[0,3,1,""],build_posterior_mean_var:[0,3,1,""],build_prior_mean_var:[0,3,1,""],child:[0,4,1,""]},"gptf.distributed.Reduction":{build_log_likelihood:[0,3,1,""],build_prior_mean_var:[0,3,1,""]},"gptf.distributed.gPoEReduction":{build_posterior_mean_var:[0,3,1,""]},"gptf.distributed.rBCMReduction":{build_posterior_mean_var:[0,3,1,""]},"gptf.gpr":{GPR:[0,2,1,""]},"gptf.gpr.GPR":{build_log_likelihood:[0,3,1,""],build_posterior_mean_var:[0,3,1,""],build_prior_mean_var:[0,3,1,""],inputs:[0,4,1,""],kernel:[0,4,1,""],likelihood:[0,4,1,""],meanfunc:[0,4,1,""],values:[0,4,1,""]},gptf:{core:[1,0,0,"-"],distributed:[0,0,0,"-"],gpr:[0,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"],"5":["py","exception","Python exception"],"6":["py","staticmethod","Python static method"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method","4":"py:attribute","5":"py:exception","6":"py:staticmethod"},terms:{"000e":1,"0_1":1,"0th":[0,1],"0x7fb127c74710":0,"1_1":1,"1st":[0,1],"abstract":1,"break":1,"case":1,"class":[0,1],"const":1,"default":[0,1],"else":1,"final":0,"function":[0,1],"import":[0,1],"int":[0,1],"long":1,"new":[0,1],"return":[0,1],"static":1,"super":1,"true":[0,1],"var":[0,1],"while":1,__call__:1,__dict__:1,__enter__:1,__exit__:1,__init__:1,__main__:1,__name__:1,__new__:1,_children:1,_dtype:1,_new_:1,_nullcontextwrapped__wrap:1,_nullcontextwrapper__wrap:1,_set_par:1,_truek:1,_unregister_child:1,abc:1,abcdefg:0,about:1,abov:1,absolut:1,access:1,account:1,acquir:1,across:1,action:1,activ:1,active_dim:1,add:[0,1],add_child:1,addit:1,addition:[0,1],additiv:1,advis:1,after:1,alia:1,alias:1,all:[0,1],allow:1,along:1,alpha:1,alreadi:1,also:1,alwai:[0,1],ancestor:1,ani:1,announcesess:1,anoth:1,ansi:1,any:1,anyth:1,appear:0,append:1,appli:1,applic:1,appropri:1,arang:1,architectur:0,ard:1,arg:1,argument:1,arithet:0,arrai:[0,1],array_display_length:1,array_equ:1,array_len:1,array_lik:1,articl:1,ascii:1,asscalar:1,assert:1,assig:1,assign:[0,1],assign_add:1,associ:1,assum:1,attempt:1,attribut:[0,1],attributeerror:1,attributetre:1,attributewrappedtf:1,autoflow:1,automat:1,axi:1,back:0,badparenterror:1,base:[0,1],basi:1,bayesian:0,bcm:0,bcmreduct:0,been:1,befor:1,behaviour:1,behind:1,below:1,bernoulli:1,beta:1,between:1,bfgs:1,bia:1,blankline:[0,1],block:1,bodi:1,bool:[0,1],both:[0,1],box:1,branch:0,breadth:1,breadthfirsttreeiter:1,build:[0,1],build_log_likelihood:[0,1],build_log_prior:1,build_posterior_mean_var:[0,1],build_predict:1,build_prior_mean_var:[0,1],cach:1,cache_clear:1,cache_limit:1,cache_method:1,calcul:[0,1],call:1,callabl:[0,1],callback:1,can:1,cannot:1,cao_fleet_weight:0,capabl:1,capac:1,care:1,caus:1,chang:[0,1],charact:1,check:1,child:[0,1],child_0:1,child_1:1,child_a:1,child_b:1,children:[0,1],chilren:0,choleski:1,choos:1,chunk:0,circumst:1,classinfo:1,clear:1,clear_all_ancestor_cach:1,clear_ancestor_cach:1,clear_cach:1,clear_subtree_cach:1,clear_tree_cach:1,client:1,cljkass:0,clone:1,close:[0,1],cluster:0,clusterdict:1,clusterspec:[0,1],code:[0,1],col:1,collect:1,column:1,combin:[0,1],combine_fancy_t:1,committe:0,common:0,compat:1,compil:1,complet:[0,1],comput:[0,1],computation:0,compute_k:1,compute_k_symm:1,compute_kdiag:1,compute_log_likelihood:1,compute_log_prior:1,compute_posterior_mean_cov:1,compute_posterior_mean_var:1,compute_posterior_sampl:1,compute_prior_mean_cov:1,compute_prior_mean_var:1,compute_prior_sampl:[0,1],concaten:1,conditional_mean:1,conditional_vari:1,config:1,conflict:1,connect:0,constant:[0,1],constrain:1,constraint:1,construct:[0,1],construct_t:1,constructor:1,contain:1,context:1,contractu:1,contrib:1,conveni:0,coordin:0,copi:[0,1],core:0,correct:[0,1],correctli:1,cosin:1,covari:[0,1],covrianc:1,creat:1,creation:1,current:1,dangl:1,data:[0,1],data_hold:1,data_summari:1,datahold:[0,1],deal:1,decomposit:1,decor:1,deepli:1,def:[0,1],defin:1,deg_fre:1,del:1,deleg:0,delet:1,demonstr:1,denomin:1,densiti:0,depend:1,depth:1,deriv:1,design:1,detail:1,determin:1,devic:1,device_index:1,device_typ:1,devicespec:1,diagon:1,dict:1,dictionari:1,did:1,differ:0,dimens:1,direct:1,dirti:1,disp:[0,1],displai:1,distanc:1,distributed_tree_rbcm:0,divid:[0,1],divis:[0,1],doc:1,doctest:1,document:1,doe:[0,1],doesn:1,don:1,doublereturnexampl:1,down:1,draw:1,dtype:[0,1],dtypechangeerror:1,due:1,each:[0,1],either:[0,1],element:1,els:1,empti:1,enabl:1,enter:1,entir:1,environ:1,equal:0,equal_weight:0,equival:1,error:1,escap:1,etc:1,euclid_dist:1,euclidean:1,eval:1,evalu:1,even:1,evenli:0,ever:1,everi:1,evict:1,exampl:[0,1],exce:1,except:[0,1],exception:1,execut:1,exist:1,exit:1,exp:[0,1],expand_dim:1,expect:1,expens:0,expert:0,explicitli:1,exponenti:1,express:[0,1],facil:1,factor:0,fall:0,fallback:1,fallback_nam:[0,1],fals:[0,1],fanci:1,fashion:0,fed:1,feed:1,feed_dict:1,fetch:1,fill:0,fill_cach:1,find:[0,1],finish:1,first:[0,1],fix:1,fixedparametererror:1,flag:1,flatten:1,flip:1,float32:1,float64:1,fmt:[0,1],follow:1,form:1,format:1,forward:1,found:1,framework:1,free:1,free_stat:1,freeli:1,from:[0,1],ftol:1,full:[0,1],full_cov:[0,1],func:[0,1],fuss:1,gamma:1,gauss:1,gaussian:[0,1],gener:[0,1],generalis:0,get:1,get_sess:1,getter:1,give:[0,1],given:1,global:1,good:1,gpflow:0,gpmodel:[0,1],gpoe:0,gpoereduct:0,gpu:1,gpy:0,grad_exp:1,grad_ident:1,gradient:1,gradientdescentoptim:1,graph:[0,1],grpc:0,guassian:1,hack:1,hand:1,handl:1,happen:1,harmon:0,hasattr:1,hashabl:1,have:[0,1],head:1,held:1,here:1,hermit:1,hierach:1,hierachi:1,hierarch:[0,1],hierarchi:1,higher:1,highest:1,highest_par:1,hold:1,hopefulli:1,hors:1,how:1,howev:1,html:1,id_fun:1,ident:1,identiti:1,ignor:1,implement:1,inde:1,independ:1,index:[0,1,2],indic:1,info:1,inherit:1,initi:1,initial_valu:[0,1],initialis:1,input:[0,1],insert:1,inspect:1,instal:1,instanc:1,instead:1,int32:[0,1],interactivesess:1,interfac:0,intern:1,invalidargumenterror:1,is_array_lik:1,isattrof:1,isclassof:1,isinst:1,isotrop:1,item:1,iter:[0,1],iterabl:[0,1],iterat:0,itself:1,job:[0,1],job_nam:1,just:1,kdiag:1,keep:1,kernal:1,kernel:0,keyerror:1,keyword:1,kind:1,knife:1,know:1,kwarg:1,lamb:1,laplac:1,last:[0,1],latent:[0,1],layer:0,lazili:1,leaf:1,learning_r:1,least:1,leav:1,len:[0,1],length:1,lengthscal:[0,1],let:1,level:1,leverag:1,librari:1,like:[0,1],likelihood:0,limit:1,line:1,linear:1,list:[0,1],listtre:1,localhost:1,log:[0,1],lognorm:1,logp:1,long_nam:1,look:1,loss:1,lower:1,lowest:1,lrucach:1,machin:0,made:1,mai:1,maintain:1,make:1,manag:1,mani:0,map:1,master:1,match:[0,1],matern12:1,matern32:1,matern52:1,matern:1,math:[0,1],matri:1,matric:1,matrix:[0,1],matur:1,maximis:1,maximum:1,maxit:1,mean:[0,1],meanfunc:0,meanfunct:0,mess:[0,1],messag:[0,1],method:[0,1],method_a:1,method_b:1,mighti:1,minim:1,minimis:1,model:0,more:1,most:1,mostli:0,move:1,mu_f:1,mul:0,multipl:1,multipli:0,multivariate_norm:1,must:[0,1],mutablemap:1,mutablesequ:1,myclass:1,name:[0,1],name_of:1,name_scop:1,namespac:1,ndarrai:1,ndim:1,neg:1,nest:1,nestedexampl:1,no_device:1,no_gc:1,node:[0,1],nois:0,noise_vari:0,non:1,none:[0,1],normal:[0,1],normalize_whitespace:1,notacopi:1,note:[0,1],noth:1,notimplement:1,now:1,np_backward:1,nullcontextwrapp:1,num:1,num_lat:[0,1],num_latent_funct:1,num_point:0,num_sampl:1,number:[0,1],numer:1,numpi:[0,1],numpyreturnexampl:1,nyi:[0,1],obj:1,object:[0,1],oblig:1,obtain:1,occur:1,odd:1,on_dtype_chang:1,on_session_birth:1,on_session_death:1,on_shape_chang:1,onc:1,one:1,ones_weight:0,onli:1,onto:1,op_placement_context:1,open:1,oper:0,operat:1,operror:1,opinion:0,opt:1,optim:[0,1],optimis:[0,1],optimiz:1,optimizeresult:1,order:[0,1],origin:[0,1],other:[0,1],other_child:1,otherwis:[0,1],our:[0,1],out:[0,1],output:[0,1],outsid:0,over:[0,1],overrid:1,overwrit:1,own:1,pad:0,padvalu:0,page:2,param:0,param_server_job:0,param_summari:1,paramat:[0,1],paramattribut:[0,1],paramet:[0,1],parameter:[0,1],parameteris:1,paramlist:[0,1],paremet:1,parent:1,part:1,partial:1,partiallyact:1,pass:1,path:1,per:1,perhap:1,pin:[0,1],place:1,placehold:1,placeholder_spec:1,placement:1,plain:[0,1],poe:0,poereduct:0,point:[0,1],point_dim:[0,1],poisson:1,posit:1,possibl:1,posterior:[0,1],power:[0,1],precis:0,precondit:0,predict:[0,1],predict_dens:1,predict_i:1,predict_mean_and_var:1,prefix:1,prefix_lin:1,print:[0,1],prior:[0,1],priordivisorreduct:0,process:[0,1],produc:1,product:0,properli:1,properti:1,proport:0,propos:1,protocol:0,provid:[0,1],proxi:1,proxywrappedvalu:1,purpos:1,push:1,put:1,python:1,quadratur:1,radial:1,rais:1,random:0,rang:[0,1],rank:1,rbcm:0,rbcmreduct:0,rbf:[0,1],read:1,reader:1,recent:1,recompil:[0,1],recurs:1,reduc:0,reduce_sum:1,reduct:0,refer:1,regardless:1,regress:0,relev:1,reli:1,remain:1,remov:1,renam:1,rename_output:1,repres:1,represent:1,request:1,requir:1,reset:1,reshap:1,resolv:1,resourc:1,respect:1,respons:1,result:1,retreiv:1,revers:1,robin:0,robust:0,root:1,round:[0,1],routin:1,row:1,rule:1,run:[0,1],safe:1,same:[0,1],sampl:[0,1],save:1,scalar:1,scale:1,scene:1,scipi:[0,1],scipyoptimizerinterfac:1,scope:1,search:[1,2],second:1,see:[0,1],self:1,sensibl:1,seper:[0,1],seq:1,sequenc:[0,1],server:[0,1],sess0:1,sess2:1,sess3:1,sess:1,session:[0,1],session_target:1,set:[0,1],setter:1,shallow:[0,1],shallowli:1,shape:[0,1],shapechangeerror:1,share:1,share_properti:1,should:[0,1],sibl:1,sigma:[0,1],similar:[0,1],simpl:1,simpli:1,singl:1,situat:1,size:[0,1],skip:1,slice:1,smoothli:1,sole:1,some:[0,1],some_attribut:1,someth:0,sort:1,sourc:[0,1],space:[0,1],spec:1,special:1,specif:1,specifi:[0,1],spoon:1,sqrt:0,squar:1,square_dist:1,start:1,state:[0,1],stationari:1,step:1,still:1,stolen:1,store:1,str:[0,1],string:1,strip_ansi_escape_cod:1,student_t:1,subclass:1,subtract:1,success:[0,1],successfulli:[0,1],suitabl:1,sum:[0,1],summar:1,summari:[0,1],suppos:[0,1],sure:1,surreptiti:1,symmetr:1,sync:1,syntax:1,system:1,tabl:1,take:1,target:[0,1],target_job:0,target_protocol:0,task:[0,1],task_index:1,tensor:[0,1],tensorflow:1,term:0,termin:1,test:[0,1],test_point:[0,1],test_valu:1,tf_add:1,tf_devic:1,tf_forward:1,tf_graph:[0,1],tf_method:1,tf_reduce_sum:1,tf_session_target:[0,1],tfhack:0,than:1,thei:[0,1],them:1,theta:1,thi:[0,1],thing:1,three:1,through:1,tile:1,time:1,top:1,tot:1,toward:1,traceback:1,track:1,train:[0,1],transform:0,treat:1,tree:0,tree_rbcm:0,treedata:1,treeparam:1,treerbcm:0,treewithcach:1,tupl:[0,1],turn:1,twice:1,two:[0,1],type:[0,1],ultimat:1,under:1,uniform:0,union:1,uniqu:[0,1],unless:1,unnam:[0,1],until:1,updat:1,util:0,vaguely_close_to:0,valu:[0,1],valueerror:1,var_f:1,variabl:1,varianc:[0,1],variational_expect:1,variou:1,vector:1,veri:1,versa:1,version:1,vice:1,wai:1,walk:1,want:1,warning:1,weight:0,weightfunct:0,weird:1,well:1,were:1,what:1,when:[0,1],where:[0,1],wherea:1,whether:1,which:[0,1],white:1,who:1,whose:1,wish:1,within:1,without:1,won:1,work:1,worker:[0,1],worker_job:0,would:1,wrap:1,wrappedtf:0,wrappedtfsess:1,wrappedvalu:1,wrapper:1,x_i:1,x_k:1,y_i:1,you:1,zero:[0,1]},titles:["gptf package","gptf.core package","Welcome to gptf&#8217;s documentation!","gptf"],titleterms:{content:[0,1],core:1,densiti:1,distribut:0,document:2,gpr:0,gptf:[0,1,2,3],indice:2,kernel:1,likelihood:1,meanfunct:1,model:1,modul:[0,1],packag:[0,1],param:1,submodul:[0,1],subpackag:0,tabl:2,tfhack:1,transform:1,tree:1,util:1,welcom:2,wrappedtf:1}})