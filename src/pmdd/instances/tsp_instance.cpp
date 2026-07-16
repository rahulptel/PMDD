// ----------------------------------------------------------
// TSP Instance - Implementations
// ----------------------------------------------------------

#include "tsp_instance.hpp"
#include <fstream>

//
// Read TSP instance
//
void TSPInstance::read(const char *inputfile) {
    ifstream input(inputfile);
    if (!input.is_open()) {
        cout << "Error: could not open file " << inputfile << endl;
        exit(1);
    }

    input >> n_objs;
    input >> n_cities;

    int val;

    objs.resize(n_objs);
    for (int o = 0; o < n_objs; ++o) {
        objs[o].resize(n_cities);
        for (int i = 0; i < n_cities; ++i) {
            objs[o][i].resize(n_cities);
            for (int j = 0; j < n_cities; ++j) {
                input >> val;
                objs[o][i][j] = val;
            }
        }
    }

    // cout << "\nMultiobjective TSP Instance" << endl;
    // cout << "\tnum of objectives = " << n_objs << endl;
    // cout << "\tnum of cities = " << n_cities << endl;
    // cout << endl;

    // minimize_bandwidth();
}