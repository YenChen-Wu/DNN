#include <util.h>
#include <nnet.h>
#include <data-io.h>
#include <iomanip>

main(){
	cout << "test!" << endl;

	cout << "io test" << endl;
	string dir("");
	mat dX = readFile((dir+"fbank/dev.ark").c_str());
	mat dY = lab2mat(readLabel((dir+"label/dev.lab").c_str()));
	cout << "good!" << endl;

}
