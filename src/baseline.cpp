#include <data-io.h>
#include <util.h>
#include <layer.h>
#include <nnet.h>

void readData(mat& X,mat& tX,mat& Y,mat& tY,string dir);
void tryONEbatch(mat& X, mat& Y,nnet* myNN);
void writeOutput(mat& X,mat& Y,nnet* myNN,ofstream& fout);

main(){
	// Parameter
	int batchSize = 128,epoch = 5;
	float lr = 0.0001,decay = 0.9999,var = 0.1,momentum = 0.9;
	int hidSize = 128;
	float lamda = 0; // L2-reg

	string dir("../libdnn_material/");
	ofstream fout("./myResult.csv");
	ofstream mout("./model");

	mat X,Y,tX,tY;readData(X,Y,tX,tY,dir);
	randShuffle(X,Y);
	// TODO
	// layer,dropout,maxout
	// dev
	//X.assign(X.begin(),X.begin()+500);
	//Y.assign(Y.begin(),Y.begin()+500);

	nnet myNN(tX[0].size(),hidSize,48,batchSize,lr,decay,lamda,var,momentum);
	myNN.nn_train(X,Y,tX,tY,epoch);
	writeOutput(tX,tY,&myNN,fout);
}

void readData(mat& X,mat& Y,mat& tX,mat& tY,string dir){
	cout << endl << "Hello, DNN!" << endl;
	cout << "====Read Data====" << endl;
	X = readFile((dir + "train.ark").c_str());
	tX = readFile((dir + "test.ark").c_str());
	Y = lab2mat(readLabel((dir + "train.lab").c_str()));
	tY = lab2mat(readLabel((dir + "test.lab").c_str()));
}

void writeOutput(mat& X,mat& Y,nnet* myNN,ofstream& fout){
	fout << "Id,Prediction" << endl;
	for(size_t i = 1;myNN->getBatch(X,X,i);i++){
		myNN->feedForward();
		for(size_t j = 0;j<myNN->output.size();j++){
			vector<float> tmp = myNN->output[j];
			int idx = max_element(tmp.begin(),tmp.end())-tmp.begin();
			fout << (i-1)*myNN->batchSize+j+1 << "," << idx << endl;
		}
	}
}

void tryONEbatch(mat& X, mat& Y,nnet* myNN){
	float err,best = 10000;
	myNN->getBatch(X,Y,1);
	for (size_t i = 0;i<500;i++){
		myNN->lr *= myNN->decay;
		myNN->backPropagate();
		err = myNN->get01Err();
		if(err<best){best=err;}
		cout << i << " " << myNN->get01Err() << " lr " << myNN->lr << " best " << best << "/" << myNN->batchSize << endl;
	}
}
