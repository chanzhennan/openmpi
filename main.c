#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define DEFAULT_TAG 1

typedef enum { false, true } bool;

void board_init(int **arr, int row, int col,int id);
void count_colors(int **grid,int rowwise_tiles_in_process,int grid_width, int tile_width, float termination_threshold, int n_itrs,int procId,int* coords);

int getTorusShortSide(int n);

bool finished = false;
int local_sum = 0, global_sum = 0;

int main(int argc,  char * argv[]){
    
    int myid, world_size;
    int n = atoi(argv[1]);
    int tile = atoi(argv[2]);
    float termination_threshold = (float)atoi(argv[3]);
    termination_threshold = termination_threshold/100;
    int MAX_ITRS = atoi(argv[4]);
    
    int tileWidth = n/tile, n_itrs = 0;
    int dim[2],period[2], reorder;
    int neighbor_coord[2];   //neighbours coordinates
    int coords[2] ={0};  //my local coordinates
    int NORTH,SOUTH,EAST,WEST;
    

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    
    if(argc != 5){
        if(myid==0)
            printf("Argument conflict !\nRun as \"main <gridsize> <tilesize> <threshold> <maxiterations>\"");
        MPI_Finalize();
        return 0;
    }
    
    if(n%tile!=0){
        if(myid==0)
            printf("(t) %d is not a multiple of (n) %d\n", tile,n);
        MPI_Finalize();
        return 0;
    }
    
    if(world_size==1){
        int tiles_in_process = tile;
        int rows_held_by_process = n, column_held_by_process = n;
        int **grid = malloc(rows_held_by_process * sizeof(int*));
        board_init(grid, rows_held_by_process,column_held_by_process, myid);
        
        
        while (!finished && n_itrs < MAX_ITRS){
            n_itrs++;
            
            /* red color movement */
            for (int i = 0; i < rows_held_by_process; i++){
                if (grid[i][0] == 1 && grid[i][1] == 0){
                    grid[i][0] = 4;
                    grid[i][1] = 3;
                }
                for (int j = 1; j < rows_held_by_process; j++){
                    if (grid[i][j] == 1 && (grid[i][(j+1)%rows_held_by_process] == 0)){
                        grid[i][j] = 0;
                        grid[i][(j+1)%rows_held_by_process] = 3;
                    }
                    else if (grid[i][j] == 3)
                        grid[i][j] = 1;
                }
                if (grid[i][0] == 3)
                    grid[i][0] = 1;
                else if (grid[i][0] == 4)
                    grid[i][0] = 0;
            }          /*end red movement */
            
            for (int j = 0; j < n; j++){
                if (grid[0][j] == 2 && grid[1][j] == 0){
                    grid[0][j] = 4;
                    grid[1][j] = 3;
                }
                for (int i = 1; i < n; i++){
                    if (grid[i][j] == 2 && grid[(i+1)%rows_held_by_process][j]==0){
                        grid[i][j] = 0;
                        grid[(i+1)%rows_held_by_process][j] = 3;
                    }
                    else if (grid[i][j] == 3)
                        grid[i][j] = 2;
                }
                if (grid[0][j] == 3)
                    grid[0][j] = 2;
                else if (grid[0][j] == 4)
                    grid[0][j] = 0;
            }
            count_colors(grid, tiles_in_process, n, tileWidth, termination_threshold, n_itrs,myid,coords);
        }
        
        if(!finished)
                printf("Threshold not reached.\nBYE !");
        
        MPI_Finalize();
        return 0;
    }
    
    //end procedure for 1 process
    
    //determine number of processes to use IF PROCESSES >
    int processesInUse;
    if(world_size >= tile*tile)
        processesInUse = tile*tile;
    else if(world_size<tile){
        processesInUse = world_size;
    }
    else {
        int divisor = world_size/tile;
        processesInUse = divisor*tile;
    }
    
    int color, groupRank, groupSize;
    if(myid<processesInUse)
        color = 1;
    else color = 0;
    
    /*
     *if 1 is returned, number is a prime, create 1D Torus and put 1 on COLUMN dimension
     */
 
    //make shorter side (n) of 2D Torus (nxm)
    int torus_short = getTorusShortSide(processesInUse);
    int torusLong = processesInUse/torus_short;
    
   // printf("TORUS SHORT %d\nTORUS LONG %d\n",torus_short,torus_long);
    MPI_Comm groupComm, cartComm;
        
    //Split processes by color
    MPI_Status status;
    MPI_Comm_split(MPI_COMM_WORLD, color, myid, &groupComm);
    MPI_Comm_rank(groupComm, &groupRank);
    MPI_Comm_size(groupComm, &groupSize);
    //printf("new group size %d\n",groupSize);
    
    dim[0] = torusLong;
    dim[1] = torus_short;
    reorder = 0;
    period[0] = 1;
    period[1] = 1;
    
    if(color == 1){             //forget abandoned processes till the end
        MPI_Cart_create(groupComm,2,dim,period,reorder,&cartComm);
        MPI_Cart_coords(cartComm, myid,dim[0],coords);
        

        if(groupRank==0){
            printf("\n--------------------------------\n"
                   "\tprocessor(s) in use : %d\n"
                   "\tConfiguration Below\n"
                   "--------------------------------\n",processesInUse);
            int proc = 0;
            for(int i = 0; i < dim[0]; i++){
                for(int j = 0; j<dim[1] ; j++){
                    printf("%d\t",proc++);
                }
                printf("\n");
            }
            printf("--------------------------------\n\n\n");
        }
        
        //determine rows and columns per process
        int columnsTilesInProcess, rowsTilesInProcess;          //columnTiles , (K) is the number of tiles (columnwise) in a process, likewise for rowTiles
        
        columnsTilesInProcess = tile / torus_short;
        if(groupRank % torus_short < (tile % torus_short))
            columnsTilesInProcess++;
        
        rowsTilesInProcess = tile / torusLong;
        if(groupRank < (tile%torusLong) * torus_short )
            rowsTilesInProcess++;
        
        

        int **grid = malloc(rowsTilesInProcess * tileWidth * sizeof(int*));
        board_init(grid, rowsTilesInProcess*tileWidth, columnsTilesInProcess * tileWidth, groupRank);
      
        int number_of_rows = rowsTilesInProcess * tileWidth;
        int number_of_columns = columnsTilesInProcess * tileWidth;
        
        int previous_grid_bottom [number_of_columns];       //ghost cells
        int next_grid_top [number_of_columns];              //ghost cells
        
        int previous_tile_right[number_of_rows];            //ghost cells
        int next_tile_left[number_of_rows];                 //ghost cells
        
        int buffer_left[number_of_rows];                    //buffer cells
        int buffer_right[number_of_rows];                   //buffer cells
        
        
        while (!finished && n_itrs < MAX_ITRS){
            n_itrs++;
            
            //collect left and right border cells into 1D array for easy communication
            for (int i = 0; i < number_of_rows; i++) {
                buffer_right [i] = grid[i][number_of_columns - 1]; //last column of each row
                buffer_left [i]  = grid[i][0];                      //first column of each row
            }
            
            //start red communication
            
            //determine EAST and WEST neighbors
            neighbor_coord[0] = coords[0];
            neighbor_coord[1] = coords[1] - 1;                        //change x coordinate to previous
            MPI_Cart_rank(cartComm, neighbor_coord, &WEST);
            
            neighbor_coord[0] = coords[0];
            neighbor_coord[1] = coords[1] + 1;
            MPI_Cart_rank(cartComm, neighbor_coord, &EAST);
            
            //SendTopRecvBottom
            MPI_Sendrecv(&buffer_left,number_of_rows,MPI_INT,WEST,DEFAULT_TAG,
                         &next_tile_left,number_of_rows,MPI_INT,EAST,DEFAULT_TAG,
                         cartComm, &status);
            
            //SendBottomRecvTop
            MPI_Sendrecv(&buffer_right,number_of_rows,MPI_INT,EAST,DEFAULT_TAG,
                         &previous_tile_right,number_of_rows,MPI_INT,WEST,DEFAULT_TAG,
                         cartComm, &status);
            
    
            /* red color movement */
            for (int i = 0; i < number_of_rows; i++){
                    //move in from left process
                if(previous_tile_right[i] == 1 && grid[i][0])
                    grid[i][0] = 3;
                
                if (grid[i][0] == 1 && grid[i][1] == 0){
                    grid[i][0] = 4;
                    grid[i][1] = 3;
                }
                for (int j = 1; j < number_of_columns; j++){
                    if (grid[i][j] == 1 && (j+1 <number_of_columns) && (grid[i][( j+1) ] == 0)){            //if not last column
                        grid[i][j] = 0;
                        grid[i][(j+1)] = 3;
                    }
                    else if (grid[i][j] == 3)
                        grid[i][j] = 1;
                    
                    //last column of row -> try to clear if next process cell is empty
                    else if( j+1 == number_of_columns){
                        if(grid[i][j] == 1 && next_tile_left[i] == 0)
                            grid[i][j] = 0;
                    }
                }
                
                if (grid[i][0] == 3)
                    grid[i][0] = 1;
                else if (grid[i][0] == 4)
                    grid[i][0] = 0;
                
            }                        //end for
            
            
            //start blue communication
            
            //determine NORTH and SOUTH NEIGHBORS
            neighbor_coord[0] = coords[0] - 1;
            neighbor_coord[1] = coords[1];
            MPI_Cart_rank(cartComm, neighbor_coord, &NORTH);
            
            neighbor_coord[0] = coords[0] + 1;
            neighbor_coord[1] = coords[1];
            MPI_Cart_rank(cartComm, neighbor_coord, &SOUTH);
            
            //SendTopRecvBottom
            MPI_Sendrecv(&grid[0][0],number_of_columns,MPI_INT,NORTH,DEFAULT_TAG,
                         &next_grid_top,number_of_columns,MPI_INT,SOUTH,DEFAULT_TAG,
                         cartComm, &status);
      
            //SendBottomRecvTop
            MPI_Sendrecv(&grid[number_of_rows-1][0],number_of_columns,MPI_INT,SOUTH,DEFAULT_TAG,
                         &previous_grid_bottom,number_of_columns,MPI_INT,NORTH,DEFAULT_TAG,
                         cartComm, &status);
            
            
            /* blue color movement */
            for (int j = 0; j < number_of_columns; j++){
                if(previous_grid_bottom[j] == 2 && grid[0][j] == 0)
                    grid[0][j]=3;                               //move in from previous grid
                
                if (grid[0][j] == 2 && grid[1][j] == 0){
                    grid[0][j] = 4;
                    grid[1][j] = 3;
                }
                for (int i = 1; i < number_of_rows; i++){

                    if (grid[i][j] == 2 && (i+1<number_of_rows) && grid[(i+1)][j] == 0){
                        grid[i][j] = 0;
                        grid[(i+1)][j] = 3;
                    }
                    else if (grid[i][j] == 3)
                        grid[i][j] = 2;
                    
                    if(grid[i][j]==2 && (i+1) == number_of_rows){
                        if(next_grid_top[j] == 0){
                            grid[i][j]=0;
                        }
                    }
                    
                }
                
                if (grid[0][j] == 3)
                    grid[0][j] = 2;
                else if (grid[0][j] == 4)
                    grid[0][j] = 0;
            }

            /* count the number of red and blue in each tile and check if the computation can be terminated*/

            count_colors(grid, rowsTilesInProcess, columnsTilesInProcess*tileWidth, tileWidth, termination_threshold, n_itrs,groupRank,coords);
            
            MPI_Allreduce(&local_sum,&global_sum,1,MPI_INT,MPI_SUM,cartComm);
            
            if(global_sum>0){
                finished = true;
            }
            
        }      //end while
        
        if(!finished){
            if(groupRank==0)
                printf("Threshold not reached.\nBYE !\n");
        }
    }
    
    
    MPI_Finalize();
}

void count_colors(int **grid,int rowwise_tiles_in_process,int grid_width, int tile_width, float termination_threshold, int n_itrs, int procId, int* global_coords){
    float whites, reds, blues;
    int count_row_tiles = 0;
    
    //processor coordinates in torus
    int processor_row = *(global_coords);
    int processor_column = *(global_coords+1);
    
    
    //calculate reds and blues in each single tile of (n/t) rows and columns
    while(count_row_tiles < rowwise_tiles_in_process){
        int count_column_tiles = 0;
        int tile_number = 0;
        //  printf("row tile # %d\n",count_row_tiles+1);
        
        while(count_column_tiles < (grid_width/tile_width)){        // single row of tiles
            
            //start tile
            whites = reds = blues = 0;
            for (int i = count_row_tiles*tile_width; i <  (count_row_tiles+1)*tile_width; i++) {
                for (int j = tile_number*tile_width ; j < tile_width*(tile_number+1); j++) {
                    if(grid[i][j]==1)
                        reds++;
                    else if(grid[i][j]==2)
                        blues++;
                    else if(grid[i][j]==0)
                        whites++;
                }
                
                
            }
            //printf("%d:%d ",count_row_tiles,count_column_tiles);
            //end tile
            
            float reds_val = reds/(tile_width*tile_width);
            float blues_val = blues/(tile_width*tile_width);
            
            
            
            if(reds_val>termination_threshold){
                //int global_r = processor_row  + count_row_tiles;
                //int global_c = processor_column + count_column_tiles ;
                printf("\nRed  {%.2f} >= threshold {%.2f}\nIteration %d\nLocal Tile Coordinates [%d:%d]\nOn Process %d\n",
                                                    reds_val,termination_threshold,n_itrs,count_row_tiles,count_column_tiles/*,global_r,global_c*/,procId);
                local_sum = 1;
            }
            if(blues_val>=termination_threshold){
                printf("\nBlue  {%.2f} >= threshold {%.2f}\nIteration %d\nLocal Tile Coordinates[%d:%d]\nOn Process %d\n",
                                                    blues_val,termination_threshold,n_itrs,count_row_tiles,count_column_tiles,procId);
                local_sum = 1;
            }
            
            count_column_tiles++;
            tile_number++;
            
        }
        count_row_tiles++;
    }
}

/*
 - get the size of the shortest dimension of the Torus
 - used to determine shape of the Torus
 - used to optimize number of processes in use
 */
int getTorusShortSide(int n){
    int max_check = n;
    int last_factor = 1;
    bool first_factor = true;
    
    //get highest factors of given number
    /*
     For example,   highest factors of 12 are 4 and 3, return 3
                    highest factors of 20 are 5 and 4, return 4
     */
    for (int i = 2; i < max_check; i++) {
        if(n%i==0){
            last_factor = i;
            if(first_factor){
                max_check = n/i;            //dont iterate past the higest factor
            }
        }
    }
    return last_factor;
}

void board_init(int **grid, int rows, int cols, int id){
    int seed = (int)time(NULL) + id;
    srand(seed);
    
    for(int i = 0; i < rows; i++)
        grid[i] = malloc(cols * sizeof(int));
    
    for (int row = 0; row < rows; row++){
        for (int col = 0; col < cols; col++){
            grid[row][col] = rand()%3;
        }
    }
}