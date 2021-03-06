#include <iomanip>
class nnet { 
public: 
	nnet(int i,int h,int o,int bat,float l,float d,float l2,float v,float m){
		nIn = i;
		nHid = h;
		nOut = o;

		batchSize = bat;
		lr = l;
		decay = d;
		lamda = l2;
		var = v;
		momentum = m;

		reset();
		//layer
		//layer L1(bat,i,h,v);
		//layer L2(bat,h,o,v);

		print();
	}

	void reset(){
		w1 = randMat(nIn+1,nHid,var);
        w2 = randMat(nHid+1,nOut,var);
        momentum_w1 = zeroMat(nIn+1,nHid);
        momentum_w2 = zeroMat(nHid+1,nOut);
		best_Ein = 1;
		best_Eout = 1;
	}

	int feedForward(){
		hidden = matrixMul(addBias(input),w1);
    	output = matrixMul(addBias(sigmoid(hidden)),w2);
    	softmax(output);
		return get01Err();
		// layer
		//L1->feedForward();
	}

	void backPropagateOracle(){
		feedForward();
		for(size_t i = 0;i<input.size();i++){
			vector<float> Out(output[i]);
			vector<float> Hid(hidden[i]);
			vector<float> In(input[i]);
			In.push_back(1);
			Hid.push_back(1);

			// compute delta 2
			vector<float> delta2(nOut,0);
			for(size_t j = 0;j<nOut;j++){
				// Cross Entropy
				delta2[j] = (Out[j]-label[i][j]);
			}
			// compute delta 1
			vector<float> delta1(nHid,0);
			for(size_t j = 0;j<nHid;j++){
				for(size_t k = 0;k<nOut;k++){
					delta1[j] += delta2[k]*w2[j][k]*Hid[j]*(1-Hid[j]);
				}
			}
			// update w1
			for(size_t k = 0;k<nIn+1;k++){
				for(size_t j = 0;j<nHid;j++){
					float grad = lr*delta1[j]*In[k];
					
					//momentum_w1[k][j] = momentum_w1[k][j]*momentum + grad;
					w1[k][j] -= grad;
				}
			}
			// update w2
			for(size_t j = 0;j<nHid+1;j++){
				for(size_t k = 0;k<nOut;k++){
					float grad =  lr*delta2[k]*Hid[j];
					//momentum_w2[j][k] = momentum_w2[j][k]*momentum + grad;
					w2[j][k] -= grad;
				}
			}
		}
		lr *= decay;
	}

	void backPropagate(){
		feedForward();
		//float max_g1,max_g2;

		for(size_t i = 0;i<input.size();i++){
			
			vector<float> Out(output[i]);
			vector<float> Hid(hidden[i]);
			vector<float> In(input[i]);
			In.push_back(1);
			Hid.push_back(1);

			// compute delta 2
			vector<float> delta2(nOut,0);
			for(size_t j = 0;j<nOut;j++){
				// L2
				//for(size_t k = 0;k<nOut;k++){
				//  float tmp = (j==k)? 1:0;
				//  delta2[j] += (label[i][k]-Out[k])*Out[j]*(tmp-Out[k]);
				//}
				// Cross Entropy
				delta2[j] = (Out[j]-label[i][j]);
			}
			// compute delta 1
			vector<float> delta1(nHid,0);
			for(size_t j = 0;j<nHid;j++){
				for(size_t k = 0;k<nOut;k++){
					delta1[j] += delta2[k]*w2[j][k]*Hid[j]*(1-Hid[j]);
				}
			}
			// update w1
			for(size_t k = 0;k<nIn+1;k++){
				for(size_t j = 0;j<nHid;j++){
					//if( abs(lr*delta1[j]*In[k]) > max_g1) max_g1=abs(lr*delta1[j]*In[k]);
					float grad = lr*delta1[j]*In[k];
					momentum_w1[k][j] = momentum_w1[k][j]*momentum + grad;
					w1[k][j] -= lamda*w1[k][j] + momentum_w1[k][j];
					//w1[k][j] -= grad + lamda*w1[k][j];
				}
			}
			// update w2
			for(size_t j = 0;j<nHid+1;j++){
				for(size_t k = 0;k<nOut;k++){
					//if( abs(lr*delta2[k]*Hid[j]) > max_g2) max_g2=abs(lr*delta2[k]*Hid[j]);
					float grad =  lr*delta2[k]*Hid[j];
					momentum_w2[j][k] = momentum_w2[j][k]*momentum + grad;
					w2[j][k] -= lamda*w2[j][k] + momentum_w2[j][k];
					//w2[j][k] -= grad + lamda*w2[j][k];
					//cout << lr << " " << delta2[k] << " " << In[j] << " " << Hid[j] << " " << lr*delta2[k]*Hid[j] << endl;
				}
			}
		
		}
		lr *= decay;
		// gradient explosion
		//cerr <<"\n\n"<< "max_g1 " << max_g1 << " max_g2 " << max_g2 ;
		//feedForward();
		//cout << "Error_before = " << Ea << " diff " << Ea - getErr() << endl;
	}


	void print(){
	    cout << "====Network====" << endl;
    	cout << "Input Node	"<< nIn << endl;
    	cout << "Hidden Node	" << nHid << endl;
    	cout << "Output Node	" << nOut << endl << endl;
		cout << "BatchSize	" << batchSize << endl;
		cout << "Learning Rate	" << lr << endl;
		cout << "Learn Decay	" << decay << endl;
		cout << "L2-regularize	" << lamda << endl;
		cout << "Init Variance	" << var << endl;
		cout << "Momentum	" << momentum << endl;
	}

	// get
	int get01Err(){return get01Error(label,output);} 	
	float getErr(){return getError(label,output);}

	bool getBatch(mat& X,mat& Y,int i){
		if(batchSize*(i-1)>X.size()){return false;}
	    mat::iterator it = X.begin();
	    mat::iterator it2 = Y.begin();
	
    	if(batchSize*i>X.size()){
    	    input =mat (it+(i-1)*batchSize,X.end());
    	    label = mat (it2+(i-1)*batchSize,Y.end());
    	}
    	else{
    	    input =mat (it+(i-1)*batchSize,it+(i)*batchSize);
    	    label = mat (it2+(i-1)*batchSize,it2+(i)*batchSize);
    	}
		return true;
	}

	void getBestModel(){w1 = good_w1;w2 = good_w2;}
	void saveModel(){good_w1 = w1;good_w2 = w2;}
	void writeModel(ofstream& out){
		//TODO
	}

	float nn_predict(mat& X,mat& Y){
   		int num = X.size(),err = 0;
    	for(size_t i = 1;getBatch(X,Y,i);i++){
    	    err += feedForward();
    	}
    	return (float)err/num;
	}
	
	void nn_train(mat& X,mat& Y,mat& tX,mat& tY,int nEpoch){
    	int nBatch = ceil((float)X.size()/batchSize);
    	cout << endl << "====Start Training====" << endl;

    	for(size_t epoch = 0; epoch<nEpoch; epoch++){
        	time_t tstart = time(0);
        	for(size_t i = 1;getBatch(X,Y,i);i++){
        	    if (i%1==0) {cerr << "\rBackpropagate Batch " << i << "/" << nBatch;}
        	    backPropagate();
	    	}
	
    		float Ein,Eout;
	    	Ein = nn_predict(X,Y);if(Ein<best_Ein){best_Ein=Ein;}
	    	Eout = nn_predict(tX,tY);if(Eout<best_Eout){best_Eout=Eout;saveModel();}

	    	cout << "\r===== Epoch " << epoch+1 << " =====";
	    	cout << "\t\tIt took "<< difftime(time(0), tstart) <<" second(s).\t";
	        cout <<  setprecision(4) << "E_in " << Ein<<"("<<best_Ein<<")"<< "\tE_out " << Eout<<"("<<best_Eout<<")"<< endl;

			if(best_Ein == 0.0)break; //
	    }
	    getBestModel();
	}
	
	void printOutput(ofstream& out){
		for(size_t i = 0;i<output.size();i++){
			for(size_t j = 0;j<output[i].size();j++)
	        	out << output[i][j] << endl;
    	}
	}

	void setTargetW(const mat& w_1,const mat& w_2){
		target_w1 = w_1;
		target_w2 = w_2;
	}

	mat input;
	mat hidden;
	mat output;
	mat label;

//private:
	int nIn;
	int nHid;
	int nOut;

	mat w1;
	mat w2;
	mat good_w1;
	mat good_w2;
	mat momentum_w1;
	mat momentum_w2;

	vector<int> nodes;
	vector<mat> w,good_w,momentum_w;

	int batchSize;
	float lr,decay,lamda,var,momentum;
	float best_Ein,best_Eout;

	mat target_w1;
	mat target_w2;
};
