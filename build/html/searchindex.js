Search.setIndex({envversion:50,filenames:["gptf","gptf.core","index","modules","public_api"],objects:{"":{gptf:[0,1,0,"-"]},"gptf.DataHolder":{feed_dict:[4,0,1,""],tensor:[4,0,1,""]},"gptf.GPModel":{build_posterior_mean_var:[4,3,1,""],build_prior_mean_var:[4,3,1,""],compute_posterior_mean_cov:[4,3,1,""],compute_posterior_mean_var:[4,3,1,""],compute_posterior_samples:[4,3,1,""],compute_prior_mean_cov:[4,3,1,""],compute_prior_mean_var:[4,3,1,""],compute_prior_samples:[4,3,1,""],predict_density:[4,3,1,""],predict_y:[4,3,1,""]},"gptf.Model":{build_log_likelihood:[4,3,1,""],build_log_prior:[4,3,1,""],compute_log_likelihood:[4,3,1,""],compute_log_prior:[4,3,1,""],optimize:[4,3,1,""]},"gptf.Param":{clear_cache:[4,3,1,""],feed_dict:[4,0,1,""],fixed:[4,0,1,""],free_state:[4,0,1,""],initializer:[4,0,1,""],on_session_birth:[4,3,1,""],on_session_death:[4,3,1,""],tensor:[4,0,1,""],transform:[4,0,1,""]},"gptf.ParamAttributes":{__setattr__:[4,3,1,""]},"gptf.Parameterized":{ARRAY_DISPLAY_LENGTH:[4,0,1,""],NO_DEVICE:[4,0,1,""],children:[4,0,1,""],clear_ancestor_caches:[4,3,1,""],clear_cache:[4,3,1,""],clear_subtree_caches:[4,3,1,""],clear_tree_caches:[4,3,1,""],copy:[4,3,1,""],data_holders:[4,0,1,""],data_summary:[4,3,1,""],feed_dict:[4,0,1,""],fixed:[4,0,1,""],get_session:[4,3,1,""],highest_parent:[4,0,1,""],long_name:[4,0,1,""],name:[4,0,1,""],name_of:[4,3,1,""],on_session_birth:[4,3,1,""],on_session_death:[4,3,1,""],op_placement_context:[4,3,1,""],param_summary:[4,3,1,""],params:[4,0,1,""],parent:[4,0,1,""],summary:[4,3,1,""],tf_device:[4,0,1,""],tf_graph:[4,0,1,""],tf_session_target:[4,0,1,""]},"gptf.core":{densities:[1,1,0,"-"],kernels:[1,1,0,"-"],likelihoods:[1,1,0,"-"],meanfunctions:[1,1,0,"-"],models:[1,1,0,"-"],params:[1,1,0,"-"],tfhacks:[1,1,0,"-"],transforms:[1,1,0,"-"],trees:[1,1,0,"-"],utils:[1,1,0,"-"],wrappedtf:[1,1,0,"-"]},"gptf.core.densities":{bernoulli:[1,4,1,""],beta:[1,4,1,""],exponential:[1,4,1,""],gamma:[1,4,1,""],gaussian:[1,4,1,""],laplace:[1,4,1,""],lognormal:[1,4,1,""],multivariate_normal:[1,4,1,""],poisson:[1,4,1,""],student_t:[1,4,1,""]},"gptf.core.kernels":{Absolute:[1,2,1,""],Additive:[1,2,1,""],Bias:[1,0,1,""],Constant:[1,2,1,""],Cosine:[1,2,1,""],Divisive:[1,2,1,""],Exponential:[1,2,1,""],Kernel:[1,2,1,""],Matern12:[1,2,1,""],Matern32:[1,2,1,""],Matern52:[1,2,1,""],Multiplicative:[1,2,1,""],Negative:[1,2,1,""],PartiallyActive:[1,2,1,""],RBF:[1,2,1,""],Static:[1,2,1,""],Stationary:[1,2,1,""],White:[1,2,1,""]},"gptf.core.kernels.Absolute":{K:[1,3,1,""],Kdiag:[1,3,1,""]},"gptf.core.kernels.Additive":{K:[1,3,1,""],Kdiag:[1,3,1,""]},"gptf.core.kernels.Constant":{K:[1,3,1,""]},"gptf.core.kernels.Divisive":{K:[1,3,1,""],Kdiag:[1,3,1,""]},"gptf.core.kernels.Kernel":{K:[1,3,1,""],Kdiag:[1,3,1,""],compute_K:[1,3,1,""],compute_K_symm:[1,3,1,""],compute_Kdiag:[1,3,1,""]},"gptf.core.kernels.Multiplicative":{K:[1,3,1,""],Kdiag:[1,3,1,""]},"gptf.core.kernels.Negative":{K:[1,3,1,""],Kdiag:[1,3,1,""]},"gptf.core.kernels.PartiallyActive":{K:[1,3,1,""],Kdiag:[1,3,1,""],active_dims:[1,0,1,""],wrapped:[1,0,1,""]},"gptf.core.kernels.Static":{Kdiag:[1,3,1,""],variance:[1,0,1,""]},"gptf.core.kernels.Stationary":{K:[1,3,1,""],Kdiag:[1,3,1,""],euclid_dist:[1,3,1,""],lengthscales:[1,0,1,""],square_dist:[1,3,1,""],variance:[1,0,1,""]},"gptf.core.kernels.White":{K:[1,3,1,""]},"gptf.core.likelihoods":{Gaussian:[1,2,1,""],Likelihood:[1,2,1,""]},"gptf.core.likelihoods.Gaussian":{conditional_mean:[1,3,1,""],conditional_variance:[1,3,1,""],logp:[1,3,1,""],predict_density:[1,3,1,""],predict_mean_and_var:[1,3,1,""],variational_expectations:[1,3,1,""]},"gptf.core.likelihoods.Likelihood":{conditional_mean:[1,3,1,""],conditional_variance:[1,3,1,""],logp:[1,3,1,""],predict_density:[1,3,1,""],predict_mean_and_var:[1,3,1,""],variational_expectations:[1,3,1,""]},"gptf.core.meanfunctions":{Absolute:[1,2,1,""],Additive:[1,2,1,""],Constant:[1,2,1,""],Divisive:[1,2,1,""],Linear:[1,2,1,""],MeanFunction:[1,2,1,""],Multiplicative:[1,2,1,""],Negative:[1,2,1,""],Zero:[1,2,1,""]},"gptf.core.meanfunctions.Divisive":{clone:[1,3,1,""]},"gptf.core.models":{GPModel:[1,2,1,""],Model:[1,2,1,""]},"gptf.core.models.GPModel":{build_posterior_mean_var:[1,3,1,""],build_prior_mean_var:[1,3,1,""],compute_posterior_mean_cov:[1,3,1,""],compute_posterior_mean_var:[1,3,1,""],compute_posterior_samples:[1,3,1,""],compute_prior_mean_cov:[1,3,1,""],compute_prior_mean_var:[1,3,1,""],compute_prior_samples:[1,3,1,""],predict_density:[1,3,1,""],predict_y:[1,3,1,""]},"gptf.core.models.Model":{build_log_likelihood:[1,3,1,""],build_log_prior:[1,3,1,""],compute_log_likelihood:[1,3,1,""],compute_log_prior:[1,3,1,""],optimize:[1,3,1,""]},"gptf.core.params":{DTypeChangeError:[1,5,1,""],DataHolder:[1,2,1,""],FixedParameterError:[1,5,1,""],Param:[1,2,1,""],ParamAttributes:[1,2,1,""],ParamList:[1,2,1,""],Parameterised:[1,0,1,""],Parameterized:[1,2,1,""],Proxy:[1,2,1,""],ProxyWrappedValue:[1,2,1,""],ShapeChangeError:[1,5,1,""],WrappedValue:[1,2,1,""],autoflow:[1,4,1,""],no_gc:[1,4,1,""],share_properties:[1,4,1,""]},"gptf.core.params.DataHolder":{feed_dict:[1,0,1,""],on_shape_change:[1,0,1,""],tensor:[1,0,1,""],value:[1,0,1,""]},"gptf.core.params.Param":{clear_cache:[1,3,1,""],feed_dict:[1,0,1,""],fixed:[1,0,1,""],free_state:[1,0,1,""],initializer:[1,0,1,""],on_session_birth:[1,3,1,""],on_session_death:[1,3,1,""],tensor:[1,0,1,""],transform:[1,0,1,""],value:[1,0,1,""]},"gptf.core.params.Parameterized":{ARRAY_DISPLAY_LENGTH:[1,0,1,""],data_holders:[1,0,1,""],data_summary:[1,3,1,""],feed_dict:[1,0,1,""],fixed:[1,0,1,""],param_summary:[1,3,1,""],params:[1,0,1,""],summary:[1,3,1,""]},"gptf.core.params.Proxy":{Shared:[1,2,1,""],copy:[1,3,1,""]},"gptf.core.params.ProxyWrappedValue":{cache:[1,0,1,""],clear_all_ancestor_caches:[1,3,1,""],on_dtype_change:[1,0,1,""],on_shape_change:[1,0,1,""],tf_device:[1,0,1,""],tf_graph:[1,0,1,""],tf_session_target:[1,0,1,""]},"gptf.core.params.WrappedValue":{on_dtype_change:[1,0,1,""],on_shape_change:[1,0,1,""],value:[1,0,1,""]},"gptf.core.tfhacks":{eye:[1,4,1,""]},"gptf.core.transforms":{Exp:[1,2,1,""],Identity:[1,2,1,""],Transform:[1,2,1,""]},"gptf.core.transforms.Exp":{np_backward:[1,3,1,""],tf_forward:[1,3,1,""]},"gptf.core.transforms.Identity":{np_backward:[1,6,1,""],tf_forward:[1,6,1,""]},"gptf.core.transforms.Transform":{np_backward:[1,3,1,""],tf_forward:[1,3,1,""]},"gptf.core.trees":{AttributeTree:[1,2,1,""],BadParentError:[1,5,1,""],BreadthFirstTreeIterator:[1,2,1,""],Leaf:[1,2,1,""],ListTree:[1,2,1,""],Tree:[1,2,1,""],TreeWithCache:[1,2,1,""],cache_method:[1,4,1,""]},"gptf.core.trees.AttributeTree":{children:[1,0,1,""],copy:[1,3,1,""],fallback_name:[1,0,1,""],name_of:[1,3,1,""]},"gptf.core.trees.Leaf":{children:[1,0,1,""],copy:[1,3,1,""]},"gptf.core.trees.ListTree":{children:[1,0,1,""],copy:[1,3,1,""],insert:[1,3,1,""],name_of:[1,3,1,""]},"gptf.core.trees.Tree":{children:[1,0,1,""],copy:[1,3,1,""],highest_parent:[1,0,1,""],long_name:[1,0,1,""],name:[1,0,1,""],name_of:[1,3,1,""],parent:[1,0,1,""]},"gptf.core.trees.TreeWithCache":{cache:[1,0,1,""],clear_ancestor_caches:[1,3,1,""],clear_cache:[1,3,1,""],clear_subtree_caches:[1,3,1,""],clear_tree_caches:[1,3,1,""],copy:[1,3,1,""]},"gptf.core.utils":{LRUCache:[1,2,1,""],combine_fancy_tables:[1,4,1,""],construct_table:[1,4,1,""],flip:[1,4,1,""],is_array_like:[1,4,1,""],isattrof:[1,4,1,""],isclassof:[1,4,1,""],prefix_lines:[1,4,1,""],strip_ansi_escape_codes:[1,4,1,""],unique:[1,4,1,""]},"gptf.core.wrappedtf":{NullContextWrapper:[1,2,1,""],WrappedTF:[1,2,1,""],WrappedTFSession:[1,2,1,""],tf_method:[1,4,1,""]},"gptf.core.wrappedtf.NullContextWrapper":{_NullContextWrapped__wrapped:[1,0,1,""]},"gptf.core.wrappedtf.WrappedTF":{NO_DEVICE:[1,0,1,""],get_session:[1,3,1,""],on_session_birth:[1,3,1,""],on_session_death:[1,3,1,""],op_placement_context:[1,3,1,""],tf_device:[1,0,1,""],tf_graph:[1,0,1,""],tf_session_target:[1,0,1,""]},"gptf.core.wrappedtf.WrappedTFSession":{close:[1,3,1,""]},"gptf.distributed":{BCMReduction:[0,2,1,""],PoEReduction:[0,2,1,""],PriorDivisorReduction:[0,2,1,""],Reduction:[0,2,1,""],cao_fleet_weights:[0,4,1,""],chunks:[0,4,1,""],distributed_tree_rBCM:[0,4,1,""],equal_weights:[0,4,1,""],gPoEReduction:[0,2,1,""],ones_weights:[0,4,1,""],rBCMReduction:[0,2,1,""],tree_rBCM:[0,4,1,""]},"gptf.distributed.BCMReduction":{build_posterior_mean_var:[0,3,1,""]},"gptf.distributed.PoEReduction":{build_posterior_mean_var:[0,3,1,""]},"gptf.distributed.PriorDivisorReduction":{build_log_likelihood:[0,3,1,""],build_posterior_mean_var:[0,3,1,""],build_prior_mean_var:[0,3,1,""],child:[4,0,1,""],weightfunction:[4,0,1,""]},"gptf.distributed.Reduction":{build_log_likelihood:[0,3,1,""],build_prior_mean_var:[0,3,1,""]},"gptf.distributed.gPoEReduction":{build_posterior_mean_var:[0,3,1,""],weightfunction:[4,0,1,""]},"gptf.distributed.rBCMReduction":{build_posterior_mean_var:[0,3,1,""],weightfunction:[4,0,1,""]},"gptf.gpr":{GPR:[0,2,1,""]},"gptf.gpr.GPR":{build_log_likelihood:[0,3,1,""],build_posterior_mean_var:[0,3,1,""],build_prior_mean_var:[0,3,1,""],kernel:[4,0,1,""],likelihood:[4,0,1,""],meanfunc:[4,0,1,""]},DataHolder:{feed_dict:[4,0,1,""],on_shape_change:[4,0,1,""],tensor:[4,0,1,""],value:[4,0,1,""]},Param:{feed_dict:[4,0,1,""],fixed:[4,0,1,""],free_state:[4,0,1,""],tensor:[4,0,1,""],transform:[4,0,1,""],value:[4,0,1,""]},Parameterized:{data_holders:[4,0,1,""],feed_dict:[4,0,1,""],fixed:[4,0,1,""],params:[4,0,1,""]},gptf:{DataHolder:[4,2,1,""],GPModel:[4,2,1,""],Model:[4,2,1,""],Param:[4,2,1,""],ParamAttributes:[4,2,1,""],ParamList:[4,2,1,""],Parameterized:[4,2,1,""],core:[1,1,0,"-"],distributed:[0,1,0,"-"],gpr:[0,1,0,"-"],tf_method:[4,4,1,""]}},objnames:{"0":["py","attribute","Python attribute"],"1":["py","module","Python module"],"2":["py","class","Python class"],"3":["py","method","Python method"],"4":["py","function","Python function"],"5":["py","exception","Python exception"],"6":["py","staticmethod","Python static method"]},objtypes:{"0":"py:attribute","1":"py:module","2":"py:class","3":"py:method","4":"py:function","5":"py:exception","6":"py:staticmethod"},terms:{"000e":[1,4],"0_1":[1,4],"0x7f0485151f60":[0,4],"1_1":[1,4],"abstract":[1,4],"break":[1,4],"case":[1,4],"class":[0,1,2],"const":[1,4],"default":[0,1,4],"else":[1,4],"final":[0,4],"function":[0,1,4],"import":[0,1,4],"int":[0,1,4],"long":[1,4],"new":[0,1,4],"public":2,"return":[0,1,4],"static":[1,4],"super":[1,4],"true":[0,1,4],"var":[1,4],"while":1,__call__:[1,4],__dict__:[1,4],__enter__:[1,4],__exit__:[1,4],__init__:[1,4],__main__:[1,4],__name__:[1,4],__new__:[1,4],__setattr__:4,_children:[1,4],_dtype:[1,4],_nullcontextwrapped__wrap:1,_nullcontextwrapper__wrap:[1,4],_set_par:[1,4],_truek:1,_unregister_child:[1,4],abc:[1,4],abcdefg:[0,4],about:[1,4],abov:[1,4],absolut:[1,4],access:[1,4],account:[1,4],acquir:[1,4],across:[1,4],action:[1,4],activ:1,active_dim:1,add:[0,1,4],add_child:[1,4],addit:[1,4],addition:[0,1,4],additiv:[1,4],advis:[1,4],after:[1,4],alexand:2,alia:[1,4],alias:[1,4],all:[0,1,4],allow:[1,4],along:1,alpha:[1,4],alreadi:[1,4],also:[1,4],alwai:[0,1,4],ancestor:[1,4],ani:[1,4],announcesess:[1,4],anoth:1,ansi:[1,4],any:1,anyth:[1,4],api:2,appear:[0,4],append:[1,4],appli:[1,4],applic:1,appropri:[1,4],arang:1,architectur:[0,4],ard:1,arg:[1,4],argument:[1,4],arithet:[0,4],arrai:[0,1,4],array_display_length:[1,4],array_equ:[1,4],array_len:[1,4],array_lik:[1,4],articl:1,ascii:[1,4],asscalar:[1,4],assert:[1,4],assig:[1,4],assign:[0,1,4],assign_add:[1,4],associ:[1,4],assum:[1,4],attempt:[1,4],attribut:[0,1,4],attributeerror:[1,4],attributetre:[1,4],attributewrappedtf:[1,4],autoflow:[1,2],automat:[1,4],avail:4,axi:1,back:[0,4],badparenterror:[1,4],base:[0,1,2,4],basi:1,bayesian:[0,4],bcm:[0,4],bcmreduct:[0,4],been:[1,4],befor:[1,4],behaviour:1,behind:[1,4],below:[1,4],bernoulli:[1,4],beta:[1,4],between:[1,4],bfgs:[1,4],bia:1,blankline:[0,1,4],block:[1,4],bodi:1,bool:[0,1,4],both:[0,1,4],box:[1,4],branch:[0,4],breadth:1,breadthfirsttreeiter:1,build:[0,1,2,4],build_log_likelihood:[0,1,4],build_log_prior:[1,4],build_posterior_mean_var:[0,1,4],build_predict:[1,4],build_prior_mean_var:[0,1,4],cach:[1,4],cache_clear:1,cache_limit:[1,4],cache_method:1,calcul:[0,1,4],call:[1,4],callabl:[0,1,4],callback:[1,4],can:[1,4],cannot:[1,4],cao_fleet_weight:[0,4],capabl:[1,4],capac:1,care:[1,4],caus:[1,4],chang:[1,4],charact:[1,4],check:1,child:[0,1,4],child_0:[1,4],child_1:[1,4],child_a:1,child_b:1,children:[0,1,4],chilren:[0,4],choleski:[1,4],choos:[1,4],chunk:[0,4],circumst:[1,4],classinfo:1,clear:[1,4],clear_all_ancestor_cach:1,clear_ancestor_cach:[1,4],clear_cach:[1,4],clear_subtree_cach:[1,4],clear_tree_cach:[1,4],client:1,cljkass:[0,4],clone:[1,4],close:[0,1,4],cluster:[0,4],clusterdict:[1,4],clusterspec:[0,1,4],code:[0,1,4],col:1,collect:1,column:[1,4],combin:[0,1,4],combine_fancy_t:1,committe:[0,4],common:[0,4],compat:[1,4],compil:[1,4],complet:[0,1,4],comput:[0,1,4],computation:[0,4],compute_k:1,compute_k_symm:1,compute_kdiag:1,compute_log_likelihood:[1,4],compute_log_prior:[1,4],compute_posterior_mean_cov:[1,4],compute_posterior_mean_var:[1,4],compute_posterior_sampl:[1,4],compute_prior_mean_cov:[1,4],compute_prior_mean_var:[1,4],compute_prior_sampl:[0,1,4],concaten:[1,4],conditional_mean:[1,4],conditional_vari:[1,4],config:1,conflict:[1,4],connect:[0,4],constant:[1,4],constrain:[1,4],constraint:[1,4],construct:[0,1,4],construct_t:1,constructor:[1,4],contain:[1,4],context:[1,4],contractu:1,contrib:[1,4],conveni:[0,4],coordin:[0,4],copi:[0,1,4],core:0,correct:[0,1,4],correctli:[1,4],cosin:1,covari:[0,1,4],covrianc:[1,4],creat:[1,4],creation:[1,4],current:[1,4],dangl:1,data:[0,1,4],data_hold:[1,4],data_summari:[1,4],datahold:[1,4],deal:1,decomposit:[1,4],decor:[1,4],deepli:1,def:[0,1,4],defin:[1,4],deg_fre:[1,4],del:1,deleg:[0,4],delet:1,demonstr:[1,4],denomin:[1,4],densiti:0,depend:[1,4],depth:[1,4],deriv:[1,4],design:[1,4],detail:[1,4],determin:1,devic:[1,4],device_index:[1,4],device_typ:[1,4],devicespec:[1,4],diagon:1,dict:[1,4],dictionari:[1,4],did:[1,4],differ:[0,4],dimens:[1,4],direct:[1,4],dirti:[1,4],disp:[0,1,4],displai:[1,4],distanc:1,distributed_tree_rbcm:[0,4],divid:[0,1,4],divis:[0,1,4],doc:[1,4],doctest:[1,4],document:1,doe:[0,1,4],doesn:[1,4],don:[1,4],doublereturnexampl:[1,4],down:[1,4],draw:[1,4],dtype:[0,1,4],dtypechangeerror:1,due:[1,4],each:[0,1,4],either:[0,1,4],element:[1,4],els:[1,4],empti:[1,4],enabl:[1,4],enter:1,entir:1,environ:[1,4],equal:[0,4],equal_weight:[0,4],equival:[1,4],error:[1,4],escap:[1,4],etc:[1,4],euclid_dist:1,euclidean:1,eval:[1,4],evalu:[1,4],even:[1,4],evenli:[0,4],ever:[1,4],everi:[0,1,4],evict:1,exampl:[0,1,4],exce:1,except:[0,1,4],exception:1,execut:[1,4],exist:[1,4],exit:1,exp:[0,1,4],expand_dim:[1,4],expect:[1,4],expens:[0,4],expert:[0,4],explicitli:[1,4],exponenti:[1,4],express:[0,1,4],facil:[1,4],factor:[0,4],fall:[0,4],fallback:1,fallback_nam:[0,1,4],fals:[0,1,4],fanci:[1,4],fashion:[0,4],fed:[1,4],feed:[1,4],feed_dict:[1,4],fetch:1,fill:[0,4],fill_cach:[1,4],find:[0,1,4],finish:[1,4],first:[0,1,4],fix:[1,4],fixedparametererror:[1,4],flag:[1,4],flatten:[1,4],flip:1,float32:[1,4],float64:[1,4],fmt:[0,1,4],follow:[1,4],form:[1,4],format:[1,4],forward:[1,4],found:[1,4],framework:[1,4],free:[1,4],free_stat:[1,4],freeli:[1,4],from:[0,1,4],ftol:[1,4],full:[0,1,4],full_cov:[0,1,4],func:[1,4],fuss:[1,4],gamma:[1,4],gauss:[1,4],gaussian:[0,1,4],gener:[0,1,4],generalis:[0,4],get:[1,4],get_sess:[1,4],getter:1,give:[0,1,4],given:[1,4],global:[1,4],good:[1,4],gpflow:[0,2],gpmodel:[0,1,4],gpoe:[0,4],gpoereduct:[0,4],gpu:[1,4],gpy:0,grad_exp:[1,4],grad_ident:[1,4],gradient:[1,4],gradientdescentoptim:[1,4],graph:[0,1,4],grpc:[0,4],guassian:[1,2,4],hack:1,hand:[1,4],handl:1,happen:[1,4],harmon:[0,4],hasattr:1,hashabl:1,have:[0,1,4],head:1,heirarchi:[1,4],held:[1,4],henman:2,here:[1,4],hermit:[1,4],hierach:[1,4],hierachi:[1,4],hierarch:[0,1,4],hierarchi:[1,4],higher:1,highest:[1,4],highest_par:[1,4],hold:[1,4],hopefulli:1,hors:1,how:[1,2,4],howev:[1,4],html:[1,4],id_fun:1,ident:[1,4],identiti:[1,4],ignor:[1,4],implement:[1,4],inde:1,independ:[1,4],index:[0,1,2,4],indic:[1,4],info:[1,4],inform:[2,4],inherit:[1,4],initi:[1,4],initial_valu:[0,1,4],initialis:[1,4],input:[0,1,4],insert:1,inspect:1,instal:[1,4],instanc:[1,4],instead:[1,4],int32:[0,1,4],interact:4,interactivesess:[1,4],interfac:0,intern:[1,4],invalidargumenterror:[1,4],is_array_lik:1,isattrof:1,isclassof:1,isinst:[1,4],isotrop:1,item:[1,4],iter:[0,1,4],iterabl:[0,1,4],iterat:[0,4],itself:1,jame:2,job:[0,1,4],job_nam:[1,4],jump:4,jupyt:2,just:[1,4],kdiag:1,keep:1,kernal:1,kernel:0,keyerror:[1,4],keyword:[1,4],kind:1,knife:[1,4],know:[1,4],kwarg:[1,4],lamb:[1,4],laplac:[1,4],last:[0,1,4],latent:[0,1,4],layer:[0,4],lazili:[1,4],leaf:1,learning_r:[1,4],least:1,leav:[1,4],len:[0,1,4],length:[1,4],lengthscal:[0,1,4],let:[1,4],level:1,leverag:[1,4],librari:[1,2],like:[0,1,4],likelihood:0,limit:[1,4],line:1,linear:[1,4],list:[0,1,4],listtre:1,localhost:[1,4],log:[0,1,4],lognorm:[1,4],logp:[1,4],long_nam:[1,4],look:[1,4],loss:[1,4],lower:[1,4],lowest:1,lrucach:1,machin:[0,4],made:[1,4],mai:[1,4],maintain:[1,4],make:1,manag:1,mani:[0,4],map:[1,4],master:[1,4],match:[0,1,4],matern12:1,matern32:1,matern52:1,matern:1,math:[1,4],matri:[1,4],matric:[1,4],matrix:[0,1,4],matthew:2,matur:1,maximis:[1,4],maximum:[1,4],maxit:[1,4],mean:[0,1,4],meanfunc:[0,4],meanfunct:0,mess:[0,1,4],messag:[0,1,4],method:[0,1,4],method_a:[1,4],method_b:[1,4],mighti:[1,4],minim:[1,4],minimis:[1,4],miscellan:[1,4],model:0,more:[1,4],most:[1,4],mostli:[0,4],move:[1,4],mu_f:[1,4],mul:[0,4],multipl:[1,4],multipli:[0,4],multivariate_norm:[1,4],must:[0,1,4],mutablemap:1,mutablesequ:1,myclass:[1,4],name:[0,1,4],name_of:[1,4],name_scop:[1,4],namespac:[1,4],ndarrai:[1,4],ndim:[1,4],neg:[1,4],nest:[1,4],nestedexampl:[1,4],no_device:[1,4],no_gc:1,node:[0,1,4],nois:[0,4],noise_vari:[0,4],non:1,none:[0,1,4],normal:[0,1,4],normalize_whitespace:[1,4],notacopi:[1,4],note:[0,1,4],notebook:[2,4],noth:[1,4],notimplement:[1,4],now:[1,4],np_backward:[1,4],nullcontextwrapp:[1,4],num:1,num_lat:[0,1,4],num_latent_funct:[1,4],num_point:[0,4],num_sampl:[1,4],number:[0,1,4],numer:[1,4],numpi:[0,1,4],numpyreturnexampl:[1,4],nyi:[0,1,4],obj:[1,4],object:[0,1,4],oblig:1,obtain:[1,4],occur:1,odd:[1,4],on_dtype_chang:1,on_session_birth:[1,4],on_session_death:[1,4],on_shape_chang:[1,4],onc:[1,4],one:[1,4],ones_weight:[0,4],onli:[1,4],onto:[1,4],op_placement_context:[1,4],open:[1,4],oper:[0,4],operat:1,operror:1,opinion:[0,4],opt:[1,4],optim:[0,1,4],optimis:[0,1,4],optimiz:[1,4],optimizeresult:[1,4],order:[0,1,4],origin:[0,1,4],other:[0,1,4],other_child:1,otherwis:[0,1,4],our:[0,1,4],out:[0,1,4],output:[0,1,4],outsid:[0,4],over:[0,1,4],overrid:[1,4],overwrit:[1,4],own:[1,4],pad:[0,4],padvalu:[0,4],page:2,param:0,param_server_job:[0,4],param_summari:[1,4],paramat:[0,1,4],paramattribut:[0,1,4],paramet:[0,1,4],parameter:[0,1,2],parameteris:[1,4],paramlist:[0,1,4],paremet:[1,4],parent:[1,4],part:[1,4],partial:1,partiallyact:1,pass:[1,4],path:[1,4],per:1,perhap:[1,4],pin:[0,1,4],place:[1,4],placehold:[1,4],placeholder_spec:[1,4],placement:[1,4],plai:4,plain:[0,1,4],poe:[0,4],poereduct:[0,4],point:[0,1,4],point_dim:[0,1,4],poisson:[1,4],posit:[1,4],possibl:[1,4],posterior:[0,1,4],power:[0,1,4],precis:[0,4],precondit:[0,4],predict:[0,1,4],predict_dens:[1,4],predict_i:[1,4],predict_mean_and_var:[1,4],prefix:1,prefix_lin:1,print:[0,1,4],prior:[0,1,4],priordivisorreduct:[0,4],process:[0,1,2,4],produc:[1,4],product:[0,4],properli:[1,4],properti:[1,4],proport:[0,4],propos:[1,4],protocol:[0,4],provid:[0,1,4],proxi:1,proxywrappedvalu:1,purpos:[1,4],push:[1,4],put:1,python:[1,4],quadratur:[1,4],radial:1,rais:[1,4],random:[0,4],rang:[0,1,4],rank:[1,4],rbcm:[0,4],rbcmreduct:[0,4],rbf:[0,1,4],read:[1,4],reader:[1,4],recent:[1,4],recompil:[1,4],recurs:[1,4],reduc:[0,4],reduce_sum:[1,4],reduct:[0,4],refer:[1,4],regardless:[1,4],regress:[0,4],relev:1,reli:[1,4],remain:[1,4],remov:[1,4],renam:[1,4],rename_output:[1,4],repres:[1,4],represent:[1,4],request:1,requir:1,reset:[1,4],reshap:1,resolv:[1,4],resourc:1,respect:[1,4],respons:[1,4],result:[1,4],retreiv:1,revers:1,robin:[0,4],robust:[0,4],root:[1,4],round:[0,1,4],routin:[1,4],row:[1,4],rule:[1,4],run:[0,1,4],safe:[1,4],same:[0,1,4],sampl:[0,1,4],save:[1,4],scalar:[1,4],scale:[1,4],scene:[1,4],scipi:[0,1,4],scipyoptimizerinterfac:[1,4],scope:[1,4],search:[1,2,4],second:[1,4],see:[0,1,2,4],self:[1,4],sensibl:1,seper:[0,1,4],seq:1,sequenc:[0,1,4],server:[0,1,4],sess0:[1,4],sess2:[1,4],sess3:[1,4],sess:[1,4],session:[0,1,4],session_target:[1,4],set:[0,1,4],setter:[1,4],shallow:[0,1,4],shallowli:1,shape:[0,1,4],shapechangeerror:1,share:1,share_properti:1,should:[0,1,4],sibl:1,sigma:[0,1,4],similar:[0,1,4],simpl:[1,4],simpli:1,singl:1,situat:[1,4],size:[0,1,4],skip:1,slice:1,smoothli:[1,4],sole:1,some:[0,1,4],some_attribut:1,someth:[0,4],sort:[1,4],sourc:[0,1,4],space:[0,1,4],spec:[1,4],special:[1,4],specif:[1,4],specifi:[0,1,4],spoon:[1,4],sqrt:[0,4],squar:1,square_dist:1,start:[1,4],state:[0,1,4],stationari:1,step:[1,4],still:[1,4],stolen:1,store:1,str:[0,1,4],string:[1,4],strip_ansi_escape_cod:1,student_t:[1,4],subclass:[1,4],subtract:[1,4],success:[0,1,4],successfulli:[0,1,4],suitabl:[1,4],sum:[0,1,4],summar:[1,4],summari:[0,1,4],sure:1,surreptiti:1,symmetr:1,sync:1,syntax:[1,4],system:[1,4],tabl:1,take:[1,4],target:[0,1,4],target_job:[0,4],target_protocol:[0,4],task:[0,1,4],task_index:[1,4],tensor:[0,1,4],tensorflow:[1,2,4],term:[0,4],termin:[1,4],test:[0,1,4],test_point:[0,1,4],test_valu:[1,4],tf_add:[1,4],tf_devic:[1,4],tf_forward:[1,4],tf_graph:[0,1,4],tf_method:[1,4],tf_reduce_sum:[1,4],tf_session_target:[0,1,4],tfhack:0,than:[1,4],thei:[0,1,4],them:[1,4],thi:[0,1,4],thing:[1,4],three:[1,4],through:[1,4],tile:[1,4],time:[1,4],top:[1,4],tot:[1,4],toward:[1,4],traceback:[1,4],track:1,train:[0,1,4],transform:0,treat:[1,4],tree:0,tree_rbcm:[0,4],treedata:[1,4],treeparam:[1,4],treerbcm:[0,4],treewithcach:[1,4],tupl:[0,1,4],turn:1,twice:1,two:[1,4],type:[0,1,4],ultimat:[1,4],under:[1,4],uniform:[0,4],union:[1,4],uniqu:[0,1,4],unless:[1,4],unnam:[0,1,4],until:1,updat:[1,4],util:0,vaguely_close_to:[0,4],valu:[0,1,4],valueerror:[1,4],var_f:[1,4],variabl:[1,4],varianc:[0,1,4],variational_expect:[1,4],variou:[1,4],vector:[1,4],veri:[1,4],versa:[1,4],version:[1,4],vice:[1,4],wai:[1,4],walk:1,want:[1,4],warning:1,weight:[0,4],weightfunct:[0,4],weird:[1,4],well:[1,4],were:[1,4],what:1,when:[0,1,4],where:[0,1,4],wherea:1,whether:[1,4],which:[0,1,4],white:1,who:[1,4],whose:[1,4],wish:[1,4],within:1,without:[1,4],won:[1,4],work:[1,4],worker:[0,1,4],worker_job:[0,4],would:[1,4],wrap:[1,4],wrappedtf:0,wrappedtfsess:1,wrappedvalu:[1,4],wrapper:1,x_i:[1,4],x_k:1,y_i:[1,4],you:[1,4],zero:[0,1,4]},titles:["gptf package","gptf.core package","Welcome to gptf&#8217;s documentation!","gptf","Public API"],titleterms:{"class":4,"public":4,api:4,autoflow:4,content:[0,1],core:[1,4],densiti:[1,4],distribut:[0,4],document:2,gpr:[0,4],gptf:[0,1,2,3],indice:2,kernel:[1,4],likelihood:[1,4],meanfunct:[1,4],model:[1,4],modul:[0,1,4],packag:[0,1],param:1,parameter:4,submodul:[0,1],subpackag:0,tabl:2,tfhack:1,transform:[1,4],tree:1,util:1,welcom:2,wrappedtf:1}})