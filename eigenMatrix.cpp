#include <iostream>

using namespace std;

#include <ctime>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
using namespace Eigen;

#define MATRIX_SIZE 50

int main(int argc, char **argv){
    Matrix<float,2,3> matrix_23;
    Vector3d v_3d;
    Matrix<float, 3, 1> vd_3d;
    Matrix3d matrix_33 = Matrix3d::Zero();
    // cout<<matrix_33<<endl;
    Matrix <double, Dynamic, Dynamic> matrix_dinamic;

    matrix_23 << 1,2,3,4,5,6;
    cout<<"matrix 2*3 1:6"<<endl<<matrix_23<<endl;
    // cout<<"print step-by-step"<<endl;
    // for (int i = 0; i<2; i++)
    //     for (int j=0; j<3; j++)
    //         cout<<matrix_23(i,j)<<endl;
    v_3d << 3,2,1;
    vd_3d << 4,5,6;
    Matrix<double, Dynamic, Dynamic> res = matrix_23.cast<double>() * v_3d;
    cout<<res.transpose()<<endl;

    matrix_33 = Matrix3d::Random();
    cout << "random matrix\n"<<matrix_33<<endl;
    cout<< "transpose: \n"<<matrix_33.transpose()<<endl;
    Matrix<float,100,100> test = Matrix<float,100,100>::Random();
    cout<<"sum:\n"<< test.sum()<<endl;
    cout<<"trace\n"<<test.trace()<<endl;
    cout<<"times 10\n"<<test.sum()*10<<endl;
    cout<<"det = "<<test.determinant()<<endl;

    Matrix<double,MATRIX_SIZE, MATRIX_SIZE> matrix_NN = MatrixXd::Random(MATRIX_SIZE,MATRIX_SIZE);
    matrix_NN *= matrix_NN.transpose();
    Matrix<double,MATRIX_SIZE,1> v_Nd = MatrixXd::Random(MATRIX_SIZE,1);

    clock_t time_stt = clock();
    Matrix<double,MATRIX_SIZE,1> x = matrix_NN.ldlt().solve(v_Nd);
    cout<<"time = "<<1000*(clock()- time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
    cout<<"x = \n"<<x.transpose()<<endl;



    cout<<"all done"<<endl;
}