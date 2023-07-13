#include <iostream>
#include <math.h>
#include <fstream>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;
using std::ofstream;

class Newton{
    public:
        Newton();
        void iteration_step();
    private:
        VectorXd r;
        MatrixXd A;
        MatrixXcd T;
};

Newton::Newton(){
    ifstream source_file("build/Complex_Transfer.txt");
    source_file.read(T);
}


int main(){

    return 0;
}