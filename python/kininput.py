#! /usr/bin/env python2.7
from numpy import *
import pickle
def _isnum(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

def stoichiometry(line,mol):
	linecp=line
	stoi=zeros(len(mol))
	rset=set()
	line=line.replace(' + ',' ')
	line=line.replace('=',' = ')
	line=line.split()
	eqindex=line.index('=')
	for i in range(len(line)):
		if line[i][0].isdigit() and not any([line[i][1]==x for x in [',','-']]):
			num,speci=int(line[i][0]),line[i][1:]
		else:	num,speci=1,line[i]
		if not (speci in mol):
			if speci!='=':
				print('SPECIOUS NOT FOUND IN REACTION: %s' % linecp)
			continue
		imol=mol.index(speci)
		rset.add(speci)
		if i < eqindex: stoi[imol]-=num
		elif i > eqindex: stoi[imol]+=num
	return(stoi,rset)

if __name__=='__main__':
	fname='./kindata.titan.moses05.N15.arsl.inp'
	infile=open(fname,'r')
	#### READ ELEMENTAL NAME AND ATOMIC MASS
	print('READ ELEMENTAL NAME AND ATOMIC MASS...')
	print('%4s%10s%10s' % ('No.','ATOM','MASS'))
	atom,mass=[],[]
	line=infile.readline()
	while not (line=='STOP\n'):
		line=line.split()
		for i in range(0,len(line),2):
			atom.append(line[i])
			mass.append(float(line[i+1]))
			print('%4i%10s%10.2f' % (len(atom),atom[-1],mass[-1]))
		line=infile.readline()
	natom=len(atom)
	mass=array(mass)
	#### READ MOLECULAR NAME AND COMPOSITION
	print('READ MOLECULAR NAME AND COMPOSITION...')
	print('%4s%12s' % ('No.','MOLECULE') + natom*'%4s' % tuple(atom))
	mol,cps=[],[]
	tmp=zeros(natom)
	line=infile.readline()
	while not (line=='STOP\n'):
		line=line.split()
		mol.append(line[0])
		for i in range(1,len(line),3):
			if (line[i] in atom):
				iatom=atom.index(line[i])
				tmp[iatom]=float(line[i+2])
			else: break
		print('%4i%12s' % (len(mol),mol[-1]) + natom*'%4i' % tuple(tmp))
		cps.append(tmp)
		tmp=zeros(natom)
		line=infile.readline()
	nmol=len(mol)
	cps=array(cps)
	#### READ REACTION LIST AND REACTION RATE
	print('READ REACTION LIST AND REACTION RATE...')
	print('%6s%45s%40s%40s\t%20s' % ('No.','REACTION','LOW PRESSURE RATE','HIGH PRESSURE RATE','REFERENCE'))
	react,stoi,rate,ref=[],[],[],[]
	tmp1,tmp21,tmp22,tmp3='','','',''
	it,ik,koo=1,0,zeros(33)
	koo[6],koo[7],koo[8]=1.,1.,0.6
	NewReaction=False
	line=infile.readline()
	while not (line=='STOP\n'):
		line=line.split()
		for i in range(len(line)):
			if not (_isnum(line[i]) or line[i]=='!'): tmp1+=line[i]
			else: break
		#### DEAL WITH SPECI WITHOU A BLANK
		tmp1=tmp1.replace('=',' = ')
		while it<len(tmp1):
			if tmp1[it]=='+' and tmp1[it-1]!='^':
				tmp1=tmp1[:it]+' + '+tmp1[it+1:]
				it=it+3
			else: it=it+1
		####
		tmp,rset=stoichiometry(tmp1,mol)
		# REMOVE EMPTY REACTION
		#if all(tmp==0):
		#	line=infile.readline()
		#	continue
		stoi.append(tmp)
		react.append((tmp1,rset))
		if i<len(line)-1:
			for j in range(i,len(line)):
				if not (line[j]=='!'):
					if line[j]=='>': NewReaction=True
					else:
						koo[ik]=float(line[j])
						ik=ik+1
				else: break
		else: j=i
		if NewReaction:
			tmp21,tmp22='??',''
		elif koo[3]==0:
			tmp21='%0.2E*(T/%0.1f)^%0.2f*exp(%0.1f/T)' % (koo[0],koo[6],koo[1],koo[2])
			tmp22=''
		else:
			tmp21='%0.2E*(T/%0.0f)^%0.1f*exp(%0.1f/T)' % (koo[0],koo[6],koo[1],koo[2])
			tmp22='%0.2E*(T/%0.1f)^%0.2f*exp(%0.1f/T)' % (koo[3],koo[7],koo[4],koo[5])
		rate.append((tmp21,tmp22,koo))
		for k in range(j+1,len(line)): tmp3+=line[k]+' '
		ref.append(tmp3)
		print('%6i%45s%40s%40s\t%20s' % (len(react),react[-1][0],rate[-1][0],rate[-1][1],ref[-1]))
		tmp1,tmp21,tmp22,tmp3='','','',''
		it,ik,koo=1,0,zeros(33)
		koo[6],koo[7],koo[8]=1.,1.,0.6
		NewReaction=False
		line=infile.readline()
	nreact=len(react)
	stoi=array(stoi)
	print('TOTAL NUMBER OF ATOMS:%8i' % natom)
	print('TOTAL NUMBER OF MOLECULES:%8i' % nmol)
	print('TOTAL NUMBER OF REACTIONS:%8i' % nreact)
	infile.close()
	#### OUTPUT VARIABLES TO PICKLE
	#outfile=open('./test.pk','w')
	#pickle.dump(atom,outfile)
	#pickle.dump(mass,outfile)
	#pickle.dump(mol,outfile)
	#pickle.dump(cps,outfile)
	#pickle.dump(react,outfile)
	#pickle.dump(rate,outfile)
	#pickle.dump(stoi,outfile)
	#pickle.dump(ref,outfile)
	#outfile.close()
	#### debug stoi
	#print '%4s%45s' % ('No.','Reaction')+nmol*'%10s' % tuple(mol)
	#for i in range(nreact):
	#	print '%4i%45s' % (i+1,react[i][0])+nmol*'%10i' % tuple(stoi[i,:])
	####
