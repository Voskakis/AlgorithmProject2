#include "modules/UserInputHandling.h"
#include "modules/CosineHashing.h"
#include "modules/EuclidianHashing.h"

using namespace std;

ifstream inFile; /* Input File */

ifstream qFile; /* Query File */

ofstream outFile; /* Output File */

int k=4; /* Number of LSH Functions */

int L=5; /* Number of Hash Tables  */

int N=5; /* Number of Nearest Neighbours to be found */

bool metric = 0; /* Metric to be used 0: Euclidian 1: Cosine */

int main(int argc, char *argv[])
{

	int input = user_input_handling(argc,argv); /* Handle User Input */

	if( input == -1 ){ 							/* Check if user input was OK */

		cout << "The program will now exit." << endl;

		return -1;

	}

	/* Initializion of Hash Functions and Hash table */

	long long unsigned int lines = get_number_of_lines();

	if( metric == 0 ){

		HashTable_Euclidian_Initialization(L);

		HashFunctions_Euclidian_Initialization(k,L);

		Euclidian_Hash_Tables_Finalization(L);

		for(int i=0; i < lines ;i++){

			Euclidian_Hash_from_file(i,L,k);

		}


		Euclidian_LSH_File(L,k,N);

	}

	if( metric == 1 ){

		HashTable_Cosine_Initialization(L,k);

		HashFunctions_Cosine_Initialization(L,k);

		for(int i=0; i < lines ;i++){

			Cosine_Hash_from_file(i,L,k);

		}

		Hash_Tables_Finalization(L,k);

		Cosine_LSH_File(L,k);

	}

 	outFile.close();
	inFile.close();
	qFile.close(); 

	return 0;

}