import Pyro4
ws=Pyro4.Proxy("PYRO:WinSpec@phd-exile-phys.ethz.ch:9091")  # use name server object lookup uri shortcut
print(ws.exposure_time)