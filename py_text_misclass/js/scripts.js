/*

on page_content for top links
<a href="#" onclick="myFunction('%s'); return false; "><h2>%s</h2></a>

on analysis_content for each section
<table class="table_alternate_colour" id="%s" style="display:none">

function myFunction(x) {

if (hour < 18) {
  greeting = "Good day";
} else {
  greeting = "Good evening";
}

  document.getElementById(x).style.display = "none";
}
*/

//<script>
$("a").click(function(){
 divId = $(this).attr("title");
  $(".linked-div").each(function(){
    if ($(this) == $("#"+divId)){
      $(this).show()
    }
    else{
      $(this).hide()
    }
  })
 $("#"+divId).show();
})
//</script>
