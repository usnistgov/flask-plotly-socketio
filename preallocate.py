import numpy as np

class PreallocatedArray(np.ndarray):

    def __new__(cls, input_array, info=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.info = info
        obj.length = 0
        obj.start_idx = 0
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        # print('In array_finalize:')
        # print('   self type is %s' % type(self))
        # print('   obj type is %s' % type(obj))
        if obj is None: return
        self.info = getattr(obj, 'info', None)
        #  when obj is preallocated array inherit length, otherwise object set
        #  length 0
        self.length = getattr(obj, 'length', 0)
        #  set start_idx = 0 when it is new
        self.start_idx = 0

    def __len__(self):
        return self.length

    def append_row(self, data):
        if len(self.shape) == 1:
            print(" Can not append a row")
            #  Can not append a row, don't do anything
            return
        if self.shape[0] == self.length:
            self[self.start_idx,:] = data
            self.start_idx += 1
            self.start_idx %= self.length
        else:
            # print(self.length)
            self[self.length, :] = data
            self.length += 1
            # print('done append', self.length)

    def __getitem__(self, i):
        # print(type(i), i)
        if (isinstance(i, slice)):
            if self.length < self.shape[0]:
                new_i = slice(i.start, self.length, i.step)
                return super(PreallocatedArray, self).__getitem__(new_i)
            else:
                index = np.arange(self.length)
                index += self.start_idx
                index %= self.length
                index = index[i]
                # print('index', index)
                return self[index]
        elif (isinstance(i, tuple)):
            if len(i) == 2:
                if np.issubdtype(type(i[1]), np.integer) and isinstance(i[0],
                                                                    slice):
                    # print('fetching a column', i)
                    if self.length < self.shape[0]:
                        old_slice = i[0]
                        new_slice = slice(old_slice.start, self.length,
                                          old_slice.step)
                        new_i = (new_slice, i[1])
                        array = super(PreallocatedArray, self).__getitem__(new_i)
                    else:
                        index = np.arange(self.length)
                        index += self.start_idx
                        index %= self.length
                        index = index[i[0]]
                        new_i = (index, i[1])
                        array = super(PreallocatedArray, self).__getitem__(new_i)

                    return np.array(array)
                else:
                    return super(PreallocatedArray, self).__getitem__(i)
            else:
                return super(PreallocatedArray, self).__getitem__(i)
        else:
            return super(PreallocatedArray, self).__getitem__(i)
