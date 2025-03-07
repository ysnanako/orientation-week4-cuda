#include "cuda_kernel.cuh"
using namespace std;

#define iteration 5

void ReadFile(vector<float> &a, vector<float> &b)
{
    ifstream fin;
    float temp;

    fin.open("A.txt");
    if(!fin)
    {
        cerr << "ERROR: input A.txt failed. \n";
        exit(1);
    }
    while(fin >> temp)
        a.push_back(temp);    
    fin.close();

    fin.open("B.txt");
    if(!fin)
    {
        cerr << "ERROR: input B.txt failed. \n";
        exit(1);
    }
    while(fin >> temp)
        b.push_back(temp);    
    fin.close();
}
void WriteFile(const vector<float> &a_cuda, const vector<float> &b_cuda)
{
    ofstream fout;

    fout.open("A_cuda.txt");
    if(!fout)
    {
        cerr << "ERROR: output A_cuda.txt failed. \n";
        exit(1);
    }
    for(size_t i = 0; i < a_cuda.size(); ++i)
        fout << a_cuda[i] << " ";
    fout.close();

    fout.open("B_cuda.txt");
    if(!fout)
    {
        cerr << "ERROR: output B_cuda.txt failed. \n";
        exit(1);
    }
    for(size_t i = 0; i < b_cuda.size(); ++i)
        fout << b_cuda[i] << " ";
    fout.close();
}
int main()
{
    vector<float> a, b, a_cuda, b_cuda;
    ReadFile(a, b);
    a_cuda = a;
    b_cuda = b;
    for(int i = 0; i < iteration; ++i)
        CUDA_init(a_cuda, b_cuda);
    WriteFile(a_cuda, b_cuda);
}