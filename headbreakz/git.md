#### git 폴더 만들기

```
git init
git remote add origin https://github.com/bcaitech1/p3-ims-obd-connectnet.git
```

or

```
git clone https://github.com/bcaitech1/p3-ims-obd-connectnet.git
```



__현재 branch__ : master



### branch 만들기

```
git branch <branch name> 

ex) git branch headbreakz
```



### branch 확인하기

```
git branch
```

 or

```
#로컬 branch의 마지막 commit 내용 포함
git branch -v
```

or

```
#로컬 / remote branch 정보
git branch -a
```



### branch update

현재 저장소에 등록되어 있는 모든 branch에 대해 정보를 update 시키며, 로컬 branch에 추가되는 것이 아닌,

`git branch -a` 를 통해 추가된 정보를 확인 가능하다.

```
git remote update
```



### branch 이동하기

`현재 branch에서 다른 branch로 이동할려면, 현재 최신 상태를 commit까지 진행해야한다.` 

```
git checkout <branch name>

ex) git checkout headbreakz
	git checkout master
```



__현재 branch__ : headbreakz



#### 작업 후

```
git add <files name>
git add .

git commit -m <message>

#현재 branch에 push
git push origin headbreakz
```





