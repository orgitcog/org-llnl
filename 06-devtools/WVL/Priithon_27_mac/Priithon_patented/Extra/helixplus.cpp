#include <fstream.h>
#include <stdlib.h>
#include <stdio.h>

#define IJK(i, j, k) ((i) + nx * ((j) + ny*(k)))

int nx=0;
int ny=0;
int nz=0;
int nxyz=0;


double xmin = -2.0;  // -1.4
double ymin = -2.0;
double zmin = -0.2;
double xside = 4.0;  // 2.8
double yside = 4.0;
double zside = 3.8; // 2.6


void zero(int * vol) 
{
  for (int i=0; i< nxyz; i++)
    vol[ i ] = 0;
}
void copy(int *v0, const int *v1)
{
  for (int i=0; i< nxyz; i++)
    v0[ i ] = v1[ i ];
}

void read(int *vol, const char *fn)
{
  zero(vol);

  ifstream inf(fn);
  cerr << "loading file...";
  
  while(!inf.eof())
    {
      double x,y,z;
      inf >> x >> y >> z;
      
      int i = int( nx * (x - xmin) / xside );
      int j = int( ny * (y - ymin) / yside );
      int k = int( nz * (z - zmin) / zside );
      
      if(i>=nx) { i = nx-1; cerr<< "nx"; }
      else if(i<0) { i = 0; cerr<< "x0"; }
      if(j>=ny) { j = ny-1; cerr<< "ny"; }
      else if(j<0) { j = 0; cerr<< "y0"; }
      if(k>=nz) { k = nz-1; cerr<< "nz"; }
      else if(k<0) { k = 0; cerr<< "z0"; }
      
      vol[ IJK(i,j,k) ] = 1;
    }
  cerr << "done."<< endl;
}
void write(const int *vol, const char *outfn)
{
  ofstream outf(outfn);
  
  cerr << "saving file...";
  for(int z=0;z<nz;z++)
    {
      for(int y=0;y<ny;y++)
	{
	  for(int x=0;x<nx;x++)
	    {
	      outf << vol[ IJK(x,y,z) ] << " ";
	      
	      if(!outf) {
		cerr << "write in error at x,y,z = " <<
		  x << " " << y << " " << z <<  endl;
		exit(1);
	      }
	    }
	}
      outf << endl;
    }
  cerr << "done."<< endl;
}

void grow(int * v1, int *v2)
{
  for(int z=0;z<nz;z++)
    {
      int zmin = z-1;
      if(zmin<0) zmin=0;
      int zmax = z+1;
      if(zmax==nz) zmax=nz-1;

      for(int y=0;y<ny;y++)
	{
	  int ymin = y-1;
	  if(ymin<0) ymin=0;
	  int ymax = y+1;
	  if(ymax==ny) ymax=ny-1;

	  for(int x=0;x<nx;x++)
	    {
	      int xmin = x-1;
	      if(xmin<0) xmin=0;
	      int xmax = x+1;
	      if(xmax==nx) xmax=nx-1;

	      int vox = v1[ IJK(x,y,z) ];
	      if(vox > 0) 
		{
		  for(int zz=zmin;zz<=zmax;zz++)
		  for(int yy=ymin;yy<=ymax;yy++)
		  for(int xx=xmin;xx<=xmax;xx++)
		    v2[ IJK(xx,yy,zz) ] = vox;
		}
	    }
	}
    }
}


int main(int ac, char *av[])
{
  if(ac < 5) {
    cerr << "usage: " << av[0] << " fn nx ny nz [grow=1]" << endl;
    exit(1);
   }

  int ngrow =1;
  if(ac >5) ngrow = atoi(av[5]);

  nx = atoi(av[2]);
  ny = atoi(av[3]);
  nz = atoi(av[4]);

  nxyz = nx * ny * nz;

  

  int *vol;
  int *vol2;
  vol = new int[  nxyz ];
  vol2 = new int[  nxyz ];

  read(vol, av[1]);

  char outfn[80];

  sprintf(outfn, "%s.vox0", av[1]);
  write(vol, outfn);


  for(int i=0;i<ngrow;i++)
    {
      zero(vol2);
      grow(vol, vol2);
      
      copy(vol, vol2);
    }




  sprintf(outfn, "%s.vox", av[1]);
  write(vol2, outfn);

  
  delete [] vol;
  delete [] vol2;
}
