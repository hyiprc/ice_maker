Ice Maker
=========

``Ice Maker`` generates initial configuration of ice for molecular dynamics simulations.

How to use
----------

   .. code-block:: console
   
       $ cd ice_maker
       $ python mk_iceSd.py

Notes
-----

+ ``dimer_cis.data`` and ``dimer_trans.data`` are the unit cells
+ ``possible_stacking.py`` generates ``possible_stacking_blank.json`` (LAMMPS data files under ``all_stacking``)
+ ``possible_stacking.json`` has manually entered cubic/hexagonal info
+ ``mk_iceSd.py`` uses ``possible_stacking.json`` to generate valid iceSD of specified number of layers that is also periodic
+ ``unit_IceSd_111.data`` is the output file, header has the stacking info and cubic/hexagonal composition