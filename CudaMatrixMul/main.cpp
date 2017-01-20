#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
using namespace std;

int main() {
	vector<int> A;
	ifstream macierz("macierz.txt");
	for (int i = 0; i < 36; i++) {
		string tmp;
		macierz >> tmp;
		stringstream ss;
		ss << tmp;
		int b = 0;
		ss >> b;
		A.push_back(b);
	}
	int wA = 6;
	int numberOfComputedCells = 3;
	int ty = 1;
	int tx = 1;
	int bs = 2;
	for (int i = 0; i < numberOfComputedCells; i++) {
		cout << "Zewnêtrzna pêtla "<< i << endl;
		for (int j = 0; j < numberOfComputedCells; j++) {
			cout << "Wewnêtrzna pêtla " << j << endl;
			for (int k = 0; k < wA; k++) {
				int row = ty*wA + k + i*wA*bs;
				int column = k*wA + tx + j*bs;
				cout << "Polozenie w wierszu: " << row << endl;
				cout << "Polozenie w kolumnie: " << column << endl;
				float A_d_element = A[row];
				float B_d_element = A[column];
				cout << A_d_element << "*" << B_d_element << endl;
				cout << "wynik: " << ty*wA + tx + i*wA*bs + j*bs << endl;
			}
			//offset += 6;
		}
	}
}