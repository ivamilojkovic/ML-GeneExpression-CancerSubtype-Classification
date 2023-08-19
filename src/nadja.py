from Crypto.PublicKey import RSA, DSA, ElGamal

def generate_new_key(name, email, algorithm, key_size):

    password = input("Enter password: ")
    
    if algorithm=='RSA':
        key = RSA.generate(bits=key_size)
    elif algorithm=='DSA':
        key = DSA.generate(bits=key_size)
    elif algorithm=='ElGamal':
        key = ElGamal.generate(bits=key_size)

    return key


def delete_key():
    pass

def import_private_key():
    pass

def export_private_key():
    pass