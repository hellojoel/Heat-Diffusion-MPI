#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <math.h>
#include <mpi.h>

using namespace std;


void argsParse(int argc, char* argv[], string* infname, int* timesteps, int* size_X, int* size_Y);
void getGridSize(int size_X, int size_Y, int world_size, int* gridsize_x, int* gridsize_y);
void memInit(int gridsize_x, int gridsize_y, double* heatmap[]);
void readCsv(string infname, int size_X, int rank_id, int gridsize_x, int gridsize_y, double* heatmap[]);
void passIntervalues(int rank_id, int world_size, int size_X, int size_Y, int gridsize_x, int gridsize_y, double* heatmap[]);
void jacobiStep(int gridsize_x, int gridsize_y, double* heatmap[]);
string getOutputName(int world_size, int size_X, int size_Y);
void writeCsv(int world_size, int size_X, int size_Y, int gridsize_x, int gridsize_y, double* heatmap[]);
void memClean(double* heatmap[]);
double getMinTime(double arr[], int size);
double getAvgTime(double arr[], int size);
double getMaxTime(double arr[], int size);
void printTime(double min_time, double avg_time, double max_time);
void fillFinalmap(int gridsize_x, int gridsize_y, double* heatmap[], double* finalmap[]);


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    string infname;
    int timesteps;
    int size_X;
    int size_Y;

    double wtime;
    int world_size;
    int rank_id;
    int gridsize_x, gridsize_y;
    double* timebuf = nullptr;
    double* heatmap = nullptr;
    double* finalmap = nullptr;
    double* aggrmap = nullptr;
    
    argsParse(argc, argv, &infname, &timesteps, &size_X, &size_Y);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);

    getGridSize(size_X, size_Y, world_size, &gridsize_x, &gridsize_y);

    memInit(gridsize_x+2, gridsize_y+2, &heatmap);
    readCsv(infname, size_X, rank_id, gridsize_x, gridsize_y, &heatmap);
    
    wtime = MPI_Wtime();
    
    for (int t=0; t<timesteps; t++) {
        passIntervalues(rank_id, world_size, size_X, size_Y, gridsize_x, gridsize_y, &heatmap);
        jacobiStep(gridsize_x, gridsize_y, &heatmap);
    }
    
    wtime = MPI_Wtime() - wtime;
    
    if (rank_id == 0) {
        timebuf = new double[world_size];
    }
    MPI_Gather(&wtime, 1, MPI_DOUBLE, timebuf, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank_id == 0) {
        printTime(getMinTime(timebuf, world_size), getAvgTime(timebuf, world_size), getMaxTime(timebuf, world_size));
        delete[] timebuf;
    }
    
    memInit(gridsize_x, gridsize_y, &finalmap);
    fillFinalmap(gridsize_x, gridsize_y, &heatmap, &finalmap);
    if (rank_id == 0) {
        aggrmap = new double[size_X*size_Y];
    }
    MPI_Gather(finalmap, gridsize_x*gridsize_y, MPI_DOUBLE, aggrmap, gridsize_x*gridsize_y, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank_id == 0) {
        writeCsv(world_size, size_X, size_Y, gridsize_x, gridsize_y, &aggrmap);
        delete[] aggrmap;
    }
    
    memClean(&heatmap);
    memClean(&finalmap);
    MPI_Finalize();

    return 0;
}


void argsParse(int argc, char* argv[], string* infname, int* timesteps, int* size_X, int* size_Y) {
    *infname = argv[argc-4];
    *timesteps = stoi(argv[argc-3]);
    *size_X = stoi(argv[argc-2]);
    *size_Y = stoi(argv[argc-1]);
    return;
}


void getGridSize(int size_X, int size_Y, int world_size, int* gridsize_x, int* gridsize_y) {
    int numblocks_x = 1;
    int numblocks_y = 1;
    int i = 0;
    while (numblocks_x*numblocks_y != world_size) {
        if (i%2) {
            numblocks_x *= 2;
        } else numblocks_y *= 2;
        i++;
    }
    *gridsize_x = size_X / numblocks_x;
    *gridsize_y = size_Y / numblocks_y;
    return;
}


void memInit(int gridsize_x, int gridsize_y, double* heatmap[]) {
    *heatmap = new double[(gridsize_x*gridsize_y)];

    for (int i = 0; i < gridsize_y; i++) {
        for (int j = 0; j < gridsize_x; j++) {
            (*heatmap)[i*gridsize_x+j] = 0.0;
        }
    }

    return;
}


void readCsv(string infname, int size_X, int rank_id, int gridsize_x, int gridsize_y, double* heatmap[]) {
    ifstream file(infname);
    string line;
    int blocks_X = size_X / gridsize_x;
    int x_start = (rank_id % blocks_X) * gridsize_x;
    int y_start = (rank_id / blocks_X) * gridsize_y;
    int r = 0;

    if (!file.is_open()) {
        cerr << "Error opening input file." << endl;
        exit(1);
    }

    while (getline(file, line)) {
        if (r >= y_start+gridsize_y) {
            break;
        } else if (r >= y_start) {
            stringstream ss(line);
            int c = 0;
            string value;
            while (getline(ss, value, ',')) {
                if (c >= x_start+gridsize_x) {
                    break;
                } else if (c >= x_start) {
                    int i = r-y_start+1;
                    int j = c-x_start+1;
                    (*heatmap)[i*(gridsize_x+2)+j] = stod(value);
                }
                c++;
            }
        }
        r++;
    }

    file.close();
}


double getMinTime(double arr[], int size) {
    double min_val = arr[0];
    for (int i = 1; i < size; i++) {
        min_val = (min_val < arr[i]) ? min_val : arr[i];
    }
    return min_val;
}


double getAvgTime(double arr[], int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum / (double)size;
}

double getMaxTime(double arr[], int size) {
    double max_val = arr[0];
    for (int i = 1; i < size; i++) {
        max_val = (max_val > arr[i]) ? max_val : arr[i];
    }
    return max_val;
}


void printTime(double min_time, double avg_time, double max_time) {
    cout << "TIME: Min: " << fixed << setprecision(3) << min_time << " s Avg: " << fixed << setprecision(3) << avg_time << " s Max: " << fixed << setprecision(3) << max_time << " s" << endl;
    return;
}


string getOutputName(int world_size, int size_X, int size_Y) {
    string fname = to_string(size_X) + "x" + to_string(size_Y) + "." + to_string(world_size) + "-output.csv";
    return fname;
}


void writeCsv(int world_size, int size_X, int size_Y, int gridsize_x, int gridsize_y, double* heatmap[]) {
    int blocks_X = size_X / gridsize_x;
    int blocks_Y = size_Y / gridsize_y;
    string outfname = getOutputName(world_size, size_X, size_Y);
    ofstream file(outfname);

    if (!file.is_open()) {
        cerr << "Error opening output file." << endl;
    }

    for (int b_y=0; b_y<blocks_Y; b_y++) {
        int processed_size = b_y*gridsize_y*size_X;
        for (int row=0; row<gridsize_y; row++) {
            int new_row = row*gridsize_x;
            for (int b_x=0; b_x<blocks_X; b_x++) {
                int shift = b_x*gridsize_x*gridsize_y;
                for (int i=0; i<gridsize_x; i++) {
                    int index = processed_size + new_row + shift + i;
                    file << fixed << setprecision(3) << (*heatmap)[index];
                    if (b_x*gridsize_x + i + 1 == size_X) {
                        file << endl;
                    } else file << ",";
                }
            }
        }
    }

    file.close();
}


void memClean(double* heatmap[]) {
    delete[] *heatmap;
    return;
}


void passIntervalues(int rank_id, int world_size, int size_X, int size_Y, int gridsize_x, int gridsize_y, double* heatmap[]) {
    double topsend[gridsize_x];
    double bottomsend[gridsize_x];
    double leftsend[gridsize_y];
    double rightsend[gridsize_y];

    double toprecv[gridsize_x];
    double bottomrecv[gridsize_x];
    double leftrecv[gridsize_y];
    double rightrecv[gridsize_y];

    int paddedsize_x = gridsize_x + 2;
    int paddedsize_y = gridsize_y + 2;
    int blocks_X = size_X / gridsize_x;
    int blocks_Y = size_Y / gridsize_y;
    int rank_x = rank_id % blocks_X;
    int rank_y = rank_id / blocks_X;

    int top_id = ((rank_y-1+blocks_Y) % blocks_Y) * blocks_X + rank_x;
    int bottom_id = ((rank_y+1) % blocks_Y) * blocks_X + rank_x;
    int left_id = rank_y * blocks_X + ((rank_x-1+blocks_X) % blocks_X);
    int right_id = rank_y * blocks_X + ((rank_x+1) % blocks_X);

    MPI_Request requests[8];

    for (int i=0; i<gridsize_x; i++) {
        topsend[i] = (*heatmap)[paddedsize_x+i+1];
        bottomsend[i] = (*heatmap)[(paddedsize_y-2)*paddedsize_x+i+1];
    }
    for (int i=0; i<gridsize_y; i++) {
        leftsend[i] = (*heatmap)[(i+1)*paddedsize_x+1];
        rightsend[i] = (*heatmap)[(i+2)*paddedsize_x-2];
    }

    MPI_Isend(topsend, gridsize_x, MPI_DOUBLE, top_id, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(bottomrecv, gridsize_x, MPI_DOUBLE, bottom_id, 0, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(bottomsend, gridsize_x, MPI_DOUBLE, bottom_id, 1, MPI_COMM_WORLD, &requests[2]);
    MPI_Irecv(toprecv, gridsize_x, MPI_DOUBLE, top_id, 1, MPI_COMM_WORLD, &requests[3]);
    MPI_Isend(leftsend, gridsize_y, MPI_DOUBLE, left_id, 2, MPI_COMM_WORLD, &requests[4]);
    MPI_Irecv(rightrecv, gridsize_y, MPI_DOUBLE, right_id, 2, MPI_COMM_WORLD, &requests[5]);
    MPI_Isend(rightsend, gridsize_y, MPI_DOUBLE, right_id, 3, MPI_COMM_WORLD, &requests[6]);
    MPI_Irecv(leftrecv, gridsize_y, MPI_DOUBLE, left_id, 3, MPI_COMM_WORLD, &requests[7]);

    MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);

    for (int i=0; i<gridsize_x; i++) {
        (*heatmap)[(paddedsize_y-1)*paddedsize_x+i+1] = bottomrecv[i];
        (*heatmap)[i+1] = toprecv[i];
    }
    for (int i=0; i<gridsize_y; i++) {
        (*heatmap)[(i+2)*paddedsize_x-1] = rightrecv[i];
        (*heatmap)[(i+1)*paddedsize_x] = leftrecv[i];
    }
    return;
}


void jacobiStep(int gridsize_x, int gridsize_y, double* heatmap[]) {
    double* new_heatmap;
    int padded_x = gridsize_x + 2;
    int padded_y = gridsize_y + 2;

    memInit(padded_x, padded_y, &new_heatmap);

    for (int i=1; i<gridsize_y+1; i++) {
        for (int j=1; j<gridsize_x+1; j++) {
            int prev_i = i - 1;
            int next_i = i + 1;
            int prev_j = j - 1;
            int next_j = j + 1;
            
            new_heatmap[i*padded_x+j] = ((*heatmap)[i*padded_x+j] +
                                   (*heatmap)[prev_i*padded_x+j] + 
                                   (*heatmap)[next_i*padded_x+j] + 
                                   (*heatmap)[i*padded_x+prev_j] + 
                                   (*heatmap)[i*padded_x+next_j]) * 0.2;
        }
    }

    for (int i=0; i<padded_y; i++) {
        for (int j=0; j<padded_x; j++) {
            (*heatmap)[i*padded_x+j] = new_heatmap[i*padded_x+j];
        }
    }

    memClean(&new_heatmap);
    return;
}

void fillFinalmap(int gridsize_x, int gridsize_y, double* heatmap[], double* finalmap[]) {
    for (int i=0; i<gridsize_y; i++) {
        for (int j=0; j<gridsize_x; j++) {
            (*finalmap)[i*gridsize_x+j] = (*heatmap)[(i+1)*(gridsize_x+2)+j+1];
        }
    }

    return;
}
