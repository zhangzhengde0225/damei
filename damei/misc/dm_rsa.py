import base64


class DmRsa(object):
    """这是加密Encrypt方案"""

    def __init__(self, plan='Encrypt'):
        """
        :param plan: 加密方案Encrypt或签密方案SignEncripy
        """
        # import rsa as raw_rsa
        self._raw_rsa = None
        self.length = 1024

    @property
    def raw_rsa(self):
        if not self._raw_rsa:
            import rsa as raw_rsa
            self._raw_rsa = raw_rsa
        return self._raw_rsa

    def gen_rsa(self, num=1, length=None):
        """在随机生成公钥-私钥对，存储在当前路径pubk.txt和privk.txt里"""
        length = length if length else self.length
        # rsa_length = int(length / 8)
        # enable_length = rsa_length - 11
        for i in range(num):
            pubk, privk = self.gen_pair(length=length)
            self._save(pubk=pubk, privk=privk, suffix=f'{i + 1}')

    def gen_pair(self, length):
        pub_key_obj, priv_key_obj = self.raw_rsa.newkeys(length)
        pub_key_str = pub_key_obj.save_pkcs1()
        pub_key_code = base64.standard_b64encode(pub_key_str).decode()

        priv_key_str = priv_key_obj.save_pkcs1()
        priv_key_code = base64.standard_b64encode(priv_key_str).decode()
        return pub_key_code, priv_key_code

    def _save(self, pubk, privk, suffix=''):
        with open(f'pubk{suffix}.pem', 'w') as f:
            f.write(pubk)
        with open(f'privk{suffix}.pem', 'w') as f:
            f.write(privk)

    def load_pk(self, pubk=None, privk=None):
        """加载钥匙"""
        assert pubk or privk  # 至少有一个
        assert not (pubk and privk)  # 不能同时都有

        if (pubk is not None) and (privk is not None):
            # print(privk, pubk, pubk is not None, privk is not None)
            raise KeyError(f'The input public key and private key both not None, the software dont know which to load.')
        elif pubk:
            if pubk.endswith('.pem'):
                with open(pubk, 'r') as f:
                    pubk = f.read()
            pubk = base64.standard_b64decode(pubk)  # 对公钥解密
            pubk = self.raw_rsa.PublicKey.load_pkcs1(pubk)  # 加载公钥
            # print(f'pubk2: {pubk}')
            return pubk
        elif privk:
            if privk.endswith('.pem'):
                with open(privk, 'r') as f:
                    privk = f.read()
            privk = base64.standard_b64decode(privk)  # 私钥解密
            privk = self.raw_rsa.PrivateKey.load_pkcs1(privk)  # 加载私钥
            return privk
        else:
            raise KeyError(f'Both public key and private key are None.')

    def encrypt(self, plaintext, pubk=None, privk=None, length=None):
        """
        用公钥或私钥对明文进行rsa加密 + base64加密
        :param plaintext，明文，str类型
        :param pubk: 公钥，str类型，需先解密，直接是公钥，或者是以.txt结尾的公钥文件
        :return: 密文，str类型
        """
        length = length if length else self.length
        # rsa加密
        pk = self.load_pk(pubk=pubk, privk=privk)
        rsa_length = int(length / 8)
        enable_length = rsa_length - 11
        ciphertext_list = []
        for i in range(0, len(plaintext), enable_length):
            v = plaintext[i:i + enable_length]
            val = self.raw_rsa.encrypt(v.encode(), pk)  # 加密，加密后是bytes类型
            ciphertext_list.append(val)
        ciphertext = b''.join(ciphertext_list)

        # base64加密
        ciphertext = base64.b64encode(ciphertext)
        ciphertext = ciphertext.decode('utf-8')

        return ciphertext

    def decrypt(self, ciphertext, privk=None, pubk=None, length=None):
        """
        利用私钥或公钥，对密文进行解密，对应的rsa+base64加密，解密是base64+rsa解密
        :param ciphertext: 密文，str类型
        :param privk: 私钥，str类型
        :param length: 1024
        :return: 明文，str类
        """
        # 先base64解密
        ciphertext = base64.b64decode(ciphertext.encode())

        # rsa解密
        length = length if length else self.length
        pk = self.load_pk(pubk=pubk, privk=privk)
        # print(f'pk: {pk} {type(pk)}')
        rsa_length = int(length / 8)
        plaintext_list = []
        for i in range(0, len(ciphertext), rsa_length):
            v = ciphertext[i:i + rsa_length]
            val = self.raw_rsa.decrypt(v, pk)
            plaintext_list.append(val)

        plaintext = b''.join(plaintext_list)
        plaintext = plaintext.decode('utf-8')
        return plaintext

        pass

    def sign(self, message, privk, hash_method='MD5'):
        """
        Sign a message by private key
        :param message: 消息
        :param privk: 私钥
        :param hash_method: MD5, SHA-1, SHA-256, SHA-512等
        :return: 签名，16进制的str
        """
        privk = self.load_pk(privk=privk)
        signature = self.raw_rsa.sign(message.encode('utf-8'), privk, hash_method=hash_method)
        # print('xxx', signature)
        signature = signature.hex()
        return signature

    def verify(self, message, signature, pubk):
        pubk = self.load_pk(pubk)
        # print('xx2', signature)
        # signature = signature.encode('utf-8')
        signature = bytes.fromhex(signature)
        # print('xx3', signature)
        try:
            self.raw_rsa.verify(message.encode('utf-8'), signature, pubk)
            return True
        except Exception as e:
            print(f'Verify failed, {e}')
            return False
# print(ret)
