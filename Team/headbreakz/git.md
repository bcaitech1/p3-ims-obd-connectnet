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

저장소에 등록되어 있는 모든 branch에 대해 정보를 update 시키며, 로컬 branch에 추가되는 것이 아닌,

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



### 다른 사람의 branch 등록하고 파일보기

현재의 로컬에 remote에 있는 branch를 불러와서 checkout을 한다. branch가 변경되면서, 해당 branch의 파일을 확인 할 수 있으며, `현재 폴더를 확인 할 경우` 기존에 사용했던 branch의 파일이 아닌 현재 branch 파일로 유지된다

```
git checkout -t origin/<branch name>
ex) git checkout -t origin/jongho
```



__현재 branch__ : jongho