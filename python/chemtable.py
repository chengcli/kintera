#! /usr/bin/env python3
from pylab import *

def isnumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def stoichiometry(name, mol):
    name_copy = name
    stoi = zeros(len(mol))
    name = name.replace('+',' ')
    name = name.replace('=',' = ')
    name = name.split()
    eqindex = name.index('=')
    reactant, resultant = [], []
    for i in range(len(name)):
        if name[i][0].isdigit():
            num, speci = int(name[i][0]), name[i][1:]
        else:   num, speci = 1, name[i]
        if speci in mol:
            imol = mol.index(speci)
            if i < eqindex:
                stoi[imol] -= num
                for j in range(num): reactant.append(speci)
            elif i > eqindex:
                stoi[imol] += num
                for j in range(num): resultant.append(speci)
        else:
            if speci != '=':
                print('SPECIOUS %s NOT FOUND IN REACTION: %s' % (speci, name_copy))
            continue
    return stoi, reactant, resultant

def write_chemtable(fname, atom, mass, mol, com, react, type, coeff):
    outfile = open(fname, 'w')
    outfile.write('Atomic name / mass:\n')
    outfile.write('%-6s%-10s%-10s\n' % ('Id.', 'Name', 'Mass'))
    for i in range(len(atom)):
        outfile.write('%-6i%-10s%-10.2f\n' % (i, atom[i], mass[i]))
    outfile.write('Molecular name / composition / boundary condition:\n')
    outfile.write('%-6s%-12s' % ('Id.', 'Molecule') \
            + len(atom) * '%-5s' % tuple(atom) \
            + '%-4s%-14s%-14s\n' % ('|', 'Top', 'Bottom'))
    for i in range(len(mol)):
        outfile.write('%-6i%-12s' % (i, mol[i]) \
            + len(atom) * '%-5i' % tuple(com[i]) \
            + '%-4s%-4d%-10.2E%-4d%-10.2E\n' % ('|', 0, 0, 0, 0))
    outfile.write('Reaction list / rate:\n')
    outfile.write(('%-6s%-5s%-28s' + '%-10s%-7s%-10s' * 3 + '\n') % \
            ('Id.', 'Type', 'Reaction', \
            'ko_A', 'ko_n', 'ko_E', \
            'koo_A', 'koo_n', 'koo_E',
            'kr_A', 'kr_n', 'kr_E'))
    for i in range(len(react)):
        outfile.write('%-6i%-5i%-28s' % (i, type[i], react[i]) \
                + '%-10.2E%-7.1f%-10.1f' * 3 % tuple(coeff[i]) \
                + '\n')
    outfile.write('STOP\n')
    outfile.close()

def read_chemtable(fname):
    atom, mass, mol, com, bnd, \
    react, type, coeff, stoi = [], [], [], [], [], [], [], [], []
    reactant, resultant = [], []
    infile = open(fname, 'r')
    infile.readline()
    infile.readline()
    line = infile.readline().split()
    while line[0] != 'Molecular':
        atom.append(line[1])
        mass.append(float(line[2]))
        line = infile.readline().split()
    natom = len(atom)
    infile.readline()
    line = infile.readline().split()
    while line[0] != 'Reaction':
        mol.append(line[1])
        iv = line.index('|')
        buf = map(int, line[2:iv])
        while len(buf) < natom: buf.append(0)
        com.append(buf)
        bnd.append(map(float, line[iv+1:]))
        line = infile.readline().split()
    nmol = len(mol)
    infile.readline()
    line = infile.readline().split()
    while line[0] != 'STOP':
        type.append(int(line[1]))
        react.append(line[2])
        coeff.append(map(float, line[3:12]))
        buf1, buf2, buf3 = stoichiometry(react[-1], mol)
        stoi.append(buf1)
        reactant.append(buf2)
        resultant.append(buf3)
        line = infile.readline().split()
    nreact = len(react)
    infile.close()
    # check mass balance
    atom = array(atom)
    mass = array(mass)
    mol = array(mol)
    com = array(com)
    bnd = array(bnd)
    react = array(react)
    type = array(type)
    coeff = array(coeff)
    stoi = array(stoi)
    print('checking mass balance...')
    bmat = dot(stoi, com)
    for i in range(nreact):
        unbalance = False
        for j in range(natom):
            if bmat[i, j] != 0:
                unbalance = True
                print(atom[j],)
        if unbalance:
            print(' unbalanced in reaction %d : %s' % (i, react[i]))
    print('DONE!')
    return atom, mass, mol, com, bnd, \
            react, type, coeff, reactant, resultant, stoi

def gen_chemtable_from_kininput(ifname, ofname):
    infile = open(ifname, 'r')
# read atomic name and mass:
    atom, mass = [], []
    line = infile.readline()
    while not (line == 'STOP\n'):
        line = line.split()
        for i in range(0, len(line), 2):
            atom.append(line[i])
            mass.append(float(line[i + 1]))
        line = infile.readline()
# read molecular name and mass:
    mol, com = [], []
    line = infile.readline()
    while not (line == 'STOP\n'):
        buffer = zeros(len(atom))
        line = line.split()
        line[0] = line[0].replace('aN', 'N')
        mol.append(line[0])
        for i in range(1, len(line), 3):
            if (line[i] in atom):
                iatom = atom.index(line[i])
                buffer[iatom] = float(line[i + 2])
            else: break
        com.append(buffer)
        line = infile.readline()
# reaction list and rate:
    react, type, stoi, coeff = [], [], [], []
    line = infile.readline()
    while not (line == 'STOP\n'):
        line = line.split()
        for i in range(len(line)):
            if isnumber(line[i]):
                ii = i
                break
        jj = len(line)
        for i in range(ii, len(line)):
            if line[i][0] == '!':
                jj = i
                break
        line = line[:jj]
        if len(line[ii:jj]) == 3:
            coeff.append(map(float, line[ii:ii+3]) + [0.] * 6)
        elif len(line[ii:jj]) == 6:
            coeff.append(map(float, line[ii+3:ii+6]) + map(float, line[ii:ii+3]) + [0.] * 3)
        name = reduce(lambda x, y : x + y, line[:ii])
        name = name.replace('aN', 'N')
        buffer, reactant, resultant = stoichiometry(name, mol)
        buffer = array(buffer)
        stoi.append(buffer)
        react.append(name)
        type.append(len(reactant))
        line = infile.readline()
    infile.close()
    atom = array(atom)
    mass = array(mass)
    mol = array(mol)
    com = array(com)
    react = array(react)
    type = array(type)
    coeff = array(coeff)
    write_chemtable(ofname, atom, mass, mol, com, react, type, coeff)

def filter_by_atom(ifname, ofname, atom_list, max_atom_num = -1, exclude = set()):
    atom, mass, mol, com, bnd, \
    react, type, coeff, reactant, resultant, stoi = read_chemtable(ifname)
    atom_index, mol_index, react_index = [], [], []
    req_atom_set = set(atom_list)
    for x in req_atom_set:
        atom_index.append(find(atom == x)[0])
    for i in range(len(mol)):
        atom_set = set()
        for j in range(len(atom)):
            if com[i, j] > 0:
                atom_set.add(atom[j])
        if atom_set <= req_atom_set:
            include_flag = True
            if max_atom_num != -1:
                for j, v in enumerate(atom_list):
                    if com[i, find(atom == v)[0]] > max_atom_num[j] and max_atom_num[j] != -1:
                        include_flag = False
            if mol[i] in set(exclude):
                include_flag = False
            if include_flag: mol_index.append(i)
    req_mol_set = set(mol[ix_(mol_index)])
    for i in range(len(react)):
        mol_set = set(reactant[i]).union(set(resultant[i]))
        if mol_set <= req_mol_set:
            react_index.append(i)
    req_react_set = set(react[ix_(react_index)])
    atom = atom[ix_(atom_index)]
    mass = mass[ix_(atom_index)]
    mol = mol[ix_(mol_index)]
    com = com[ix_(mol_index, atom_index)]
    bnd = bnd[ix_(mol_index)]
    react = react[ix_(react_index)]
    type = type[ix_(react_index)]
    coeff = coeff[ix_(react_index, range(9))]
    write_chemtable(ofname, atom, mass, mol, com, react, type, coeff)

def refresh(fname):
    atom, mass, mol, com, bnd, \
    react, type, coeff, reactant, resultant, stoi = read_chemtable(fname)
    # sort reaction first by type and second by name
    ind = lexsort((react, type))
    react = react[ix_(ind)]
    type = type[ix_(ind)]
    coeff = coeff[ix_(ind, range(9))]
    write_chemtable(fname, atom, mass, mol, com, react, type, coeff)

def reaction_matrix(fname):
    atom, mass, mol, com, bnd, \
    react, type, coeff, reactant, resultant, stoi = read_chemtable(fname)
    nmol, nreact = len(mol), len(react)
    kernal = zeros((nmol, nmol))
    j = 1
    for i in range(nreact):
        if stoi[i, j] > 0:
            id = find(stoi[i, :] < 0)
            if len(id) == 1 and stoi[i, id[0]] == -2:
                kernal[id[0], id[0]] = 1
            elif len(id) == 2:
                kernal[id[0], id[1]] = 1
                kernal[id[1], id[0]] = 1
        if stoi[i, j] < 0:
            id = find(stoi[i, :] < 0)
            if len(id) == 1:
                kernal[0, id[0] + 1] = -1
                kernal[id[0] + 1, 0] = -1
            else:
                kernal[id[0] + 1, id[1] + 1] = -1
                kernal[id[1] + 1, id[0] + 1] = -1
    print(kernal)


if __name__ == '__main__':
    gen_chemtable_from_kininput(
            './kindata.titan.moses05.N15.arsl.inp',
            './chemtable.txt'
            )
    filter_by_atom(
            './chemtable.txt',
            './chemtable.CH.txt',
            ['C', 'H'],
            max_atom_num = [2, -1],
            exclude = ['U', 'V', 'W', 'Y', 'RAYEAR']
            )
    refresh('./chemtable.CH.txt')
    #reaction_matrix('./chemtable.CH.txt')
